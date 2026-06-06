const GITHUB_TOKEN = "";
const API_ROOT = "https://api.github.com";
const ONE_DAY = 24 * 60 * 60 * 1000;

const headers = {
  Accept: "application/vnd.github+json",
  ...(GITHUB_TOKEN ? { Authorization: `Bearer ${GITHUB_TOKEN}` } : {}),
};

async function github(path) {
  const response = await fetch(`${API_ROOT}${path}`, { headers });

  if (!response.ok) {
    const message = response.status === 403
      ? "GitHub rate limit reached. Add a token in src/githubApi.js."
      : `GitHub API error ${response.status}`;
    throw new Error(message);
  }

  return response.json();
}

export async function loadGitHubActivity(username) {
  const user = await github(`/users/${encodeURIComponent(username)}`);
  const repos = await github(`/users/${encodeURIComponent(username)}/repos?per_page=100&sort=pushed&type=owner`);
  const activeRepos = repos.filter((repo) => !repo.fork).slice(0, 24);

  const [commitGroups, languageGroups] = await Promise.all([
    Promise.all(activeRepos.map((repo) => loadRepoCommits(repo, username))),
    Promise.all(activeRepos.map(loadRepoLanguages)),
  ]);

  const commits = commitGroups.flat();
  const languages = mergeLanguages(languageGroups);
  const dailyCommits = buildDailyCommitSeries(commits);
  const streaks = calculateStreaks(dailyCommits);

  return {
    user: structureUser(user),
    repos: activeRepos.map(structureRepo),
    commits,
    dailyCommits,
    streaks,
    languages,
    summary: {
      totalCommits: commits.length,
      scannedRepos: activeRepos.length,
      activeDays: dailyCommits.filter((day) => day.count > 0).length,
    },
  };
}

async function loadRepoCommits(repo, username) {
  try {
    const since = new Date(Date.now() - 365 * ONE_DAY).toISOString();
    const commits = await github(`/repos/${repo.full_name}/commits?author=${encodeURIComponent(username)}&since=${since}&per_page=100`);

    return commits.map((item) => ({
      sha: item.sha,
      repo: repo.name,
      date: item.commit.author?.date ?? item.commit.committer?.date,
      message: item.commit.message,
      url: item.html_url,
    })).filter((commit) => commit.date);
  } catch (error) {
    return [];
  }
}

async function loadRepoLanguages(repo) {
  try {
    return github(`/repos/${repo.full_name}/languages`);
  } catch (error) {
    return {};
  }
}

function structureUser(user) {
  return {
    login: user.login,
    name: user.name || user.login,
    avatarUrl: user.avatar_url,
    bio: user.bio || "No public bio available.",
    profileUrl: user.html_url,
    followers: user.followers,
    publicRepos: user.public_repos,
  };
}

function structureRepo(repo) {
  return {
    name: repo.name,
    fullName: repo.full_name,
    stars: repo.stargazers_count,
    forks: repo.forks_count,
    language: repo.language,
    pushedAt: repo.pushed_at,
    url: repo.html_url,
  };
}

function buildDailyCommitSeries(commits) {
  const counts = new Map();
  commits.forEach((commit) => {
    const key = toDateKey(new Date(commit.date));
    counts.set(key, (counts.get(key) || 0) + 1);
  });

  return Array.from({ length: 365 }, (_, index) => {
    const date = new Date(Date.now() - (364 - index) * ONE_DAY);
    const key = toDateKey(date);
    return { date, key, count: counts.get(key) || 0 };
  });
}

function calculateStreaks(days) {
  let current = 0;
  let best = 0;
  let running = 0;

  days.forEach((day) => {
    if (day.count > 0) {
      running += 1;
      best = Math.max(best, running);
    } else {
      running = 0;
    }
  });

  for (let index = days.length - 1; index >= 0; index -= 1) {
    if (days[index].count === 0) break;
    current += 1;
  }

  return { current, best, activeWeeks: countActiveWeeks(days) };
}

function countActiveWeeks(days) {
  const weeks = new Set();
  days.forEach((day) => {
    if (day.count > 0) weeks.add(`${day.date.getFullYear()}-${getWeekNumber(day.date)}`);
  });
  return weeks.size;
}

function getWeekNumber(date) {
  const start = new Date(date.getFullYear(), 0, 1);
  const diff = date - start + (start.getTimezoneOffset() - date.getTimezoneOffset()) * 60 * 1000;
  return Math.floor(diff / (7 * ONE_DAY));
}

function mergeLanguages(languageGroups) {
  const totals = new Map();
  languageGroups.forEach((languages) => {
    Object.entries(languages).forEach(([name, bytes]) => {
      totals.set(name, (totals.get(name) || 0) + bytes);
    });
  });

  const totalBytes = Array.from(totals.values()).reduce((sum, bytes) => sum + bytes, 0) || 1;

  return Array.from(totals, ([name, bytes]) => ({
    name,
    bytes,
    percent: Math.round((bytes / totalBytes) * 1000) / 10,
  })).sort((a, b) => b.bytes - a.bytes).slice(0, 8);
}

function toDateKey(date) {
  return date.toISOString().slice(0, 10);
}
