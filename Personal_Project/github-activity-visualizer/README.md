# GitHub Activity Visualizer

A D3-powered public GitHub activity dashboard that pulls and structures GitHub API data into commit heatmaps, streak analytics, language breakdowns, and a shareable profile card.

## Run locally

Because the app uses ES modules, serve the folder with any static server:

```bash
cd github-activity-visualizer
python -m http.server 8000
```

Open `http://localhost:8000`.

## Data pipeline

`src/githubApi.js` pulls and normalizes:

- User profile from `/users/{username}`
- Owned repositories from `/users/{username}/repos`
- Recent authored commits from `/repos/{owner}/{repo}/commits`
- Repository language bytes from `/repos/{owner}/{repo}/languages`

The exported `loadGitHubActivity(username)` returns structured data:

- `user`: display-ready profile fields
- `repos`: normalized repository metadata
- `commits`: commit records with repo, date, message, URL
- `dailyCommits`: 365-day commit count series
- `streaks`: current streak, best streak, active weeks
- `languages`: merged language percentages
- `summary`: total commits, scanned repos, active days

## Rate limits

Unauthenticated GitHub API calls are rate-limited. Add a personal access token to `GITHUB_TOKEN` in `src/githubApi.js` for higher limits.
