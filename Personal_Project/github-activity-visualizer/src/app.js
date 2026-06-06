import { loadGitHubActivity } from "./githubApi.js";
import { renderHeatmap, renderLanguageChart, renderStreakBars } from "./charts.js";

const form = document.querySelector("#search-form");
const status = document.querySelector("#status");
const usernameInput = document.querySelector("#username");
const copyCard = document.querySelector("#copy-card");
let latestActivity = null;

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const username = usernameInput.value.trim();
  if (!username) return;
  await visualize(username);
});

copyCard.addEventListener("click", async () => {
  if (!latestActivity) return;
  const { user, summary, streaks, languages } = latestActivity;
  const topLanguages = languages.slice(0, 3).map((language) => `${language.name} ${language.percent}%`).join(", ") || "No language data";
  const text = `${user.name} (@${user.login}) — ${summary.totalCommits} commits in the last year, ${streaks.current}-day current streak, ${streaks.best}-day best streak. Top languages: ${topLanguages}.`;
  await navigator.clipboard.writeText(text);
  setStatus("Profile card summary copied to clipboard.");
});

async function visualize(username) {
  setStatus(`Pulling public GitHub activity for @${username}...`);
  try {
    const activity = await loadGitHubActivity(username);
    latestActivity = activity;
    updateProfile(activity);
    renderHeatmap("#heatmap", activity.dailyCommits);
    renderStreakBars("#streak-bars", activity.streaks);
    renderLanguageChart("#language-chart", activity.languages);
    setStatus(`Structured ${activity.commits.length} commits from ${activity.summary.scannedRepos} repositories.`);
  } catch (error) {
    setStatus(error.message);
  }
}

function updateProfile(activity) {
  document.querySelector("#avatar").src = activity.user.avatarUrl;
  document.querySelector("#display-name").textContent = activity.user.name;
  document.querySelector("#bio").textContent = activity.user.bio;
  document.querySelector("#total-commits").textContent = activity.summary.totalCommits;
  document.querySelector("#current-streak").textContent = activity.streaks.current;
  document.querySelector("#best-streak").textContent = activity.streaks.best;
  document.querySelector("#repo-count").textContent = `${activity.summary.scannedRepos} repos scanned`;
}

function setStatus(message) {
  status.textContent = message;
}

visualize(usernameInput.value);
