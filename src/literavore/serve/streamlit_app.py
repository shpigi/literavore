"""Streamlit search UI for Literavore."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8000"


def make_api_request(
    endpoint: str,
    method: str = "GET",
    data: dict[str, Any] | None = None,
    timeout: int = 10,
) -> dict[str, Any] | None:
    """Make an HTTP request to the Literavore API.

    Returns the parsed JSON response, or None on failure.
    """
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, timeout=timeout)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=timeout)
        else:
            return None
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]
    except requests.exceptions.ConnectionError:
        return None
    except requests.exceptions.RequestException as exc:
        logger.warning("API request failed: %s", exc)
        return None


def check_api_available() -> dict[str, Any] | None:
    """Check API availability via /health endpoint."""
    return make_api_request("/health")


def get_available_conferences() -> list[str]:
    """Fetch list of available conferences from the API."""
    data = make_api_request("/conferences")
    if data and isinstance(data.get("conferences"), list):
        return sorted(data["conferences"])
    return []


def search_papers(
    query: str,
    top_k: int = 10,
    conference_filter: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Send a search request to the API and return results."""
    payload: dict[str, Any] = {"query": query, "top_k": top_k}
    if conference_filter:
        payload["conference_filter"] = conference_filter
    data = make_api_request("/search", method="POST", data=payload)
    if data and isinstance(data.get("results"), list):
        return data["results"]  # type: ignore[no-any-return]
    return []


def get_paper_detail(paper_id: str) -> dict[str, Any] | None:
    """Fetch full paper details from GET /papers/{id}."""
    return make_api_request(f"/papers/{paper_id}")


def format_authors(authors: Any) -> str:
    """Format an authors list into a readable string."""
    if not authors:
        return "Unknown"
    if isinstance(authors, list):
        names: list[str] = []
        for a in authors:
            if isinstance(a, dict):
                names.append(a.get("name", ""))
            elif isinstance(a, str):
                names.append(a)
        names = [n for n in names if n]
        if len(names) == 0:
            return "Unknown"
        if len(names) == 1:
            return names[0]
        if len(names) == 2:
            return f"{names[0]}, {names[1]}"
        return f"{names[0]} et al."
    if isinstance(authors, str):
        return authors
    return "Unknown"


def render_result_card(result: dict[str, Any], index: int) -> None:
    """Render a single search result as a card."""
    paper_id = result.get("id", "")
    title = result.get("title", "Untitled")
    authors = format_authors(result.get("authors"))
    venue = result.get("venue") or result.get("conference") or "Unknown"
    score = result.get("score")
    summary = result.get("summary") or result.get("abstract") or ""

    summary_excerpt = summary[:300] + "..." if len(summary) > 300 else summary

    with st.container():
        col_main, col_action = st.columns([5, 1])
        with col_main:
            st.markdown(f"**{title}**")
            st.caption(f"Authors: {authors}  |  Venue: {venue}")
            if score is not None:
                st.caption(f"Score: {score:.4f}")
            if summary_excerpt:
                st.markdown(
                    f"<small style='color:#aaa;'>{summary_excerpt}</small>",
                    unsafe_allow_html=True,
                )
        with col_action:
            if paper_id:
                if st.button("Details", key=f"detail_{index}_{paper_id}"):
                    st.session_state["selected_paper_id"] = paper_id
        st.divider()


def render_paper_detail(paper_id: str) -> None:
    """Render the full paper detail view."""
    detail = get_paper_detail(paper_id)
    if not detail:
        st.warning("Could not load paper details. The API may be unavailable.")
        return

    title = detail.get("title", "Untitled")
    authors = format_authors(detail.get("authors"))
    venue = detail.get("venue") or detail.get("conference") or "Unknown"
    abstract = detail.get("abstract") or ""
    summary = detail.get("summary") or ""
    tags = detail.get("tags") or []
    url = detail.get("url") or detail.get("openreview_url") or ""

    st.markdown(f"### {title}")
    st.caption(f"Authors: {authors}  |  Venue: {venue}")

    if url:
        st.markdown(f"[View on OpenReview]({url})")

    if abstract:
        with st.expander("Abstract", expanded=True):
            st.write(abstract)

    if summary:
        with st.expander("AI Summary", expanded=True):
            st.write(summary)

    if tags:
        if isinstance(tags, list):
            tag_str = "  ".join(f"`{t}`" for t in tags if isinstance(t, str))
        else:
            tag_str = str(tags)
        st.markdown(f"**Tags:** {tag_str}")

    if st.button("Back to results"):
        st.session_state["selected_paper_id"] = None
        st.rerun()


def build_scatter_plot(results: list[dict[str, Any]]) -> go.Figure:
    """Build a 2D scatter plot of search results.

    Uses the result score as the x-axis (rank/relevance) and
    a simple index as the y-axis. Points are colored by venue.
    """
    if not results:
        return go.Figure()

    rows = []
    for i, r in enumerate(results):
        title = r.get("title", "Untitled")
        authors = format_authors(r.get("authors"))
        venue = r.get("venue") or r.get("conference") or "Unknown"
        score = r.get("score") or 0.0
        rows.append(
            {
                "id": r.get("id", str(i)),
                "title": title,
                "authors": authors,
                "venue": venue,
                "score": score,
                "rank": i + 1,
            }
        )

    df = pd.DataFrame(rows)
    unique_venues = sorted(df["venue"].unique())

    import plotly.colors

    palettes = [
        plotly.colors.qualitative.Plotly,
        plotly.colors.qualitative.Set3,
        plotly.colors.qualitative.Bold,
    ]
    all_colors = [c for p in palettes for c in p]
    venue_color_map = {v: all_colors[i % len(all_colors)] for i, v in enumerate(unique_venues)}

    fig = go.Figure()
    for venue in unique_venues:
        vdf = df[df["venue"] == venue]
        fig.add_trace(
            go.Scatter(
                x=vdf["score"].tolist(),
                y=vdf["rank"].tolist(),
                mode="markers",
                marker=dict(size=10, color=venue_color_map[venue]),
                name=venue,
                text=vdf["title"].tolist(),
                customdata=list(zip(vdf["authors"].tolist(), vdf["venue"].tolist())),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Authors: %{customdata[0]}<br>"
                    "Venue: %{customdata[1]}<br>"
                    "Score: %{x:.4f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Search Results — Score vs Rank",
        xaxis_title="Similarity Score",
        yaxis_title="Rank",
        yaxis=dict(autorange="reversed"),
        height=400,
        showlegend=True,
        plot_bgcolor="#1E1E1E",
        paper_bgcolor="#1E1E1E",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
    )
    return fig


def render_sidebar() -> tuple[list[str], int]:
    """Render the sidebar controls and return (conference_filter, top_k)."""
    st.sidebar.title("Literavore")
    st.sidebar.markdown("Semantic search for conference papers")

    st.sidebar.markdown("---")

    conferences = get_available_conferences()
    if conferences:
        selected_conferences: list[str] = st.sidebar.multiselect(
            "Conference filter",
            options=conferences,
            default=[],
            help="Filter results to these conferences only",
        )
    else:
        selected_conferences = []
        st.sidebar.caption("No conference data available")

    top_k: int = st.sidebar.slider(
        "Results per search (top k)",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
    )

    return selected_conferences, top_k


def main() -> None:
    """Entry point for the Streamlit UI."""
    st.set_page_config(page_title="Literavore", layout="wide")

    # Sidebar
    conference_filter, top_k = render_sidebar()

    # Check if we are in paper detail mode
    selected_paper_id: str | None = st.session_state.get("selected_paper_id")
    if selected_paper_id:
        render_paper_detail(selected_paper_id)
        return

    # Main header
    st.title("Literavore — Conference Paper Search")

    # API availability check
    health = check_api_available()
    if health is None:
        st.warning(
            "The Literavore API is not reachable at http://localhost:8000. "
            "Start it with `literavore serve` to enable search."
        )
    else:
        paper_count = health.get("papers_count") or health.get("total_papers", "?")
        st.markdown(
            f"<span style='color:#00c853;'>API connected — {paper_count} papers indexed</span>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Search form
    col_input, col_button = st.columns([5, 1])
    with col_input:
        query = st.text_input(
            "Search query",
            placeholder="e.g. diffusion models for image generation",
            label_visibility="collapsed",
        )
    with col_button:
        search_clicked = st.button("Search", type="primary", use_container_width=True)

    # Perform search
    if search_clicked and query.strip():
        if health is None:
            st.error("Cannot search: API is not available.")
            return
        with st.spinner("Searching..."):
            results = search_papers(
                query.strip(),
                top_k=top_k,
                conference_filter=conference_filter if conference_filter else None,
            )
        st.session_state["last_results"] = results
        st.session_state["last_query"] = query.strip()

    results: list[dict[str, Any]] = st.session_state.get("last_results", [])
    last_query: str = st.session_state.get("last_query", "")

    if not results and not (search_clicked and query.strip()):
        st.info("Enter a search query above and click **Search** to find relevant papers.")
        return

    if results:
        st.markdown(f"### Results for: *{last_query}* ({len(results)} found)")

        # 2D scatter plot
        fig = build_scatter_plot(results)
        st.plotly_chart(fig, use_container_width=True)

        # Result cards
        for i, result in enumerate(results):
            render_result_card(result, i)
    elif search_clicked:
        st.warning("No results found. Try a different query.")


if __name__ == "__main__":
    main()
