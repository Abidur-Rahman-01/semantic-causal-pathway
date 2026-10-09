"""GitHub Model Context Protocol (MCP) Server for Antigravity.

Provides native GitHub tools over MCP Stdio transport using GitHub REST API.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from typing import Any, Optional

import requests
from mcp.server.mcpserver import MCPServer

# Initialize MCP Server
app = MCPServer("github")

API_BASE = "https://api.github.com"


def _get_headers() -> dict[str, str]:
    token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "Antigravity-GitHub-MCP/1.0",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _request(method: str, endpoint: str, **kwargs) -> Any:
    url = f"{API_BASE}/{endpoint.lstrip('/')}"
    headers = _get_headers()
    custom_headers = kwargs.pop("headers", {})
    headers.update(custom_headers)
    
    resp = requests.request(method, url, headers=headers, **kwargs)
    if resp.status_code >= 400:
        try:
            err_data = resp.json()
            msg = err_data.get("message", resp.text)
        except Exception:
            msg = resp.text
        return {"error": f"GitHub API Error ({resp.status_code}): {msg}", "status_code": resp.status_code}
    if resp.status_code == 204:
        return {"success": True, "status_code": 204}
    try:
        return resp.json()
    except Exception:
        return resp.text


@app.tool()
def get_authenticated_user() -> str:
    """Get the profile of the currently authenticated GitHub user."""
    data = _request("GET", "/user")
    return json.dumps(data, indent=2)


@app.tool()
def search_repositories(query: str, limit: int = 10) -> str:
    """Search for repositories on GitHub matching a search query.
    
    Args:
        query: Search query string (e.g. 'machine learning language:python').
        limit: Maximum number of results to return (default 10, max 100).
    """
    params = {"q": query, "per_page": min(max(1, limit), 100)}
    data = _request("GET", "/search/repositories", params=params)
    items = data.get("items", []) if isinstance(data, dict) else []
    summary = [{
        "full_name": r.get("full_name"),
        "description": r.get("description"),
        "html_url": r.get("html_url"),
        "stars": r.get("stargazers_count"),
        "language": r.get("language"),
        "private": r.get("private"),
    } for r in items]
    return json.dumps({"total_count": data.get("total_count", 0), "repositories": summary}, indent=2)


@app.tool()
def list_user_repositories(username: str = "", limit: int = 30) -> str:
    """List repositories for a specific user or the authenticated user.
    
    Args:
        username: GitHub username. If empty, lists repositories for the authenticated user.
        limit: Maximum number of repositories to return (default 30).
    """
    endpoint = f"/users/{username}/repos" if username else "/user/repos"
    params = {"per_page": min(max(1, limit), 100), "sort": "updated"}
    data = _request("GET", endpoint, params=params)
    if isinstance(data, list):
        summary = [{
            "name": r.get("name"),
            "full_name": r.get("full_name"),
            "description": r.get("description"),
            "html_url": r.get("html_url"),
            "private": r.get("private"),
            "default_branch": r.get("default_branch"),
            "fork": r.get("fork"),
            "updated_at": r.get("updated_at"),
        } for r in data]
        return json.dumps(summary, indent=2)
    return json.dumps(data, indent=2)


@app.tool()
def get_repository(owner: str, repo: str) -> str:
    """Get detailed information about a specific GitHub repository.
    
    Args:
        owner: Owner of the repository (username or org).
        repo: Repository name.
    """
    data = _request("GET", f"/repos/{owner}/{repo}")
    return json.dumps(data, indent=2)


@app.tool()
def create_repository(name: str, description: str = "", private: bool = False, auto_init: bool = True) -> str:
    """Create a new GitHub repository for the authenticated user.
    
    Args:
        name: Name of the repository.
        description: Description of the repository.
        private: Whether the repository should be private (default False).
        auto_init: Pass True to create an initial commit with empty README (default True).
    """
    payload = {
        "name": name,
        "description": description,
        "private": private,
        "auto_init": auto_init,
    }
    data = _request("POST", "/user/repos", json=payload)
    return json.dumps(data, indent=2)


@app.tool()
def get_file_contents(owner: str, repo: str, path: str, ref: str = "") -> str:
    """Get file contents from a repository.
    
    Args:
        owner: Repository owner.
        repo: Repository name.
        path: File path inside repository (e.g. 'src/main.py').
        ref: Git branch, tag, or commit SHA (optional, defaults to default branch).
    """
    params = {"ref": ref} if ref else {}
    data = _request("GET", f"/repos/{owner}/{repo}/contents/{path.lstrip('/')}", params=params)
    if isinstance(data, dict) and "content" in data and data.get("encoding") == "base64":
        try:
            decoded = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
            return json.dumps({
                "path": data.get("path"),
                "sha": data.get("sha"),
                "size": data.get("size"),
                "content": decoded,
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Failed to decode base64: {e}", "raw": data}, indent=2)
    return json.dumps(data, indent=2)


@app.tool()
def create_or_update_file(owner: str, repo: str, path: str, content: str, message: str, branch: str = "main", sha: str = "") -> str:
    """Create or update a file in a repository.
    
    Args:
        owner: Repository owner.
        repo: Repository name.
        path: File path inside repository.
        content: Text content of the file to write.
        message: Commit message.
        branch: Branch to commit to (default 'main').
        sha: The blob SHA of the file being replaced (required when updating an existing file).
    """
    encoded = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    payload = {
        "message": message,
        "content": encoded,
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha
    data = _request("PUT", f"/repos/{owner}/{repo}/contents/{path.lstrip('/')}", json=payload)
    return json.dumps(data, indent=2)


@app.tool()
def list_commits(owner: str, repo: str, sha: str = "", limit: int = 20) -> str:
    """List commits on a repository branch.
    
    Args:
        owner: Repository owner.
        repo: Repository name.
        sha: SHA or branch to start listing commits from.
        limit: Number of commits to return (default 20).
    """
    params = {"per_page": min(max(1, limit), 100)}
    if sha:
        params["sha"] = sha
    data = _request("GET", f"/repos/{owner}/{repo}/commits", params=params)
    if isinstance(data, list):
        summary = [{
            "sha": c.get("sha")[:8],
            "full_sha": c.get("sha"),
            "author": (c.get("commit") or {}).get("author", {}).get("name"),
            "date": (c.get("commit") or {}).get("author", {}).get("date"),
            "message": (c.get("commit") or {}).get("message"),
        } for c in data]
        return json.dumps(summary, indent=2)
    return json.dumps(data, indent=2)


@app.tool()
def list_issues(owner: str, repo: str, state: str = "open", limit: int = 30) -> str:
    """List issues in a repository.
    
    Args:
        owner: Repository owner.
        repo: Repository name.
        state: State of issues ('open', 'closed', 'all'; default 'open').
        limit: Max issues to return (default 30).
    """
    params = {"state": state, "per_page": min(max(1, limit), 100)}
    data = _request("GET", f"/repos/{owner}/{repo}/issues", params=params)
    if isinstance(data, list):
        summary = [{
            "number": i.get("number"),
            "title": i.get("title"),
            "state": i.get("state"),
            "user": (i.get("user") or {}).get("login"),
            "html_url": i.get("html_url"),
            "comments": i.get("comments"),
            "created_at": i.get("created_at"),
            "is_pull_request": "pull_request" in i,
        } for i in data]
        return json.dumps(summary, indent=2)
    return json.dumps(data, indent=2)


@app.tool()
def get_issue(owner: str, repo: str, issue_number: int) -> str:
    """Get details of a specific issue in a repository.
    
    Args:
        owner: Repository owner.
        repo: Repository name.
        issue_number: Issue number.
    """
    data = _request("GET", f"/repos/{owner}/{repo}/issues/{issue_number}")
    return json.dumps(data, indent=2)


@app.tool()
def create_issue(owner: str, repo: str, title: str, body: str = "", labels: list[str] = None) -> str:
    """Create a new issue in a repository.
    
    Args:
        owner: Repository owner.
        repo: Repository name.
        title: Issue title.
        body: Issue description body.
        labels: List of label strings to attach.
    """
    payload = {"title": title, "body": body}
    if labels:
        payload["labels"] = labels
    data = _request("POST", f"/repos/{owner}/{repo}/issues", json=payload)
    return json.dumps(data, indent=2)


@app.tool()
def add_issue_comment(owner: str, repo: str, issue_number: int, body: str) -> str:
    """Add a comment to an existing issue or pull request.
    
    Args:
        owner: Repository owner.
        repo: Repository name.
        issue_number: Issue or PR number.
        body: Comment body text.
    """
    payload = {"body": body}
    data = _request("POST", f"/repos/{owner}/{repo}/issues/{issue_number}/comments", json=payload)
    return json.dumps(data, indent=2)


@app.tool()
def list_pull_requests(owner: str, repo: str, state: str = "open", limit: int = 30) -> str:
    """List pull requests in a repository.
    
    Args:
        owner: Repository owner.
        repo: Repository name.
        state: State of PRs ('open', 'closed', 'all'; default 'open').
        limit: Max PRs to return (default 30).
    """
    params = {"state": state, "per_page": min(max(1, limit), 100)}
    data = _request("GET", f"/repos/{owner}/{repo}/pulls", params=params)
    if isinstance(data, list):
        summary = [{
            "number": p.get("number"),
            "title": p.get("title"),
            "state": p.get("state"),
            "user": (p.get("user") or {}).get("login"),
            "head": (p.get("head") or {}).get("ref"),
            "base": (p.get("base") or {}).get("ref"),
            "html_url": p.get("html_url"),
            "draft": p.get("draft"),
            "created_at": p.get("created_at"),
        } for p in data]
        return json.dumps(summary, indent=2)
    return json.dumps(data, indent=2)


@app.tool()
def create_pull_request(owner: str, repo: str, title: str, head: str, base: str = "main", body: str = "") -> str:
    """Create a new pull request in a repository.
    
    Args:
        owner: Repository owner.
        repo: Repository name.
        title: PR title.
        head: The name of the branch where your changes are implemented.
        base: The name of the branch you want the changes pulled into (default 'main').
        body: PR description.
    """
    payload = {"title": title, "head": head, "base": base, "body": body}
    data = _request("POST", f"/repos/{owner}/{repo}/pulls", json=payload)
    return json.dumps(data, indent=2)


@app.tool()
def search_code(query: str, limit: int = 10) -> str:
    """Search code across GitHub repositories.
    
    Args:
        query: Code search query (e.g. 'function_name repo:owner/repo').
        limit: Max results to return (default 10).
    """
    params = {"q": query, "per_page": min(max(1, limit), 100)}
    data = _request("GET", "/search/code", params=params)
    items = data.get("items", []) if isinstance(data, dict) else []
    summary = [{
        "name": item.get("name"),
        "path": item.get("path"),
        "repository": (item.get("repository") or {}).get("full_name"),
        "html_url": item.get("html_url"),
    } for item in items]
    return json.dumps({"total_count": data.get("total_count", 0), "code_results": summary}, indent=2)


def main():
    app.run(transport="stdio")


if __name__ == "__main__":
    main()
