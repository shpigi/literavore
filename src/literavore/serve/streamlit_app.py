"""Streamlit search UI for Literavore."""

from __future__ import annotations

import logging
from typing import Any

import plotly.graph_objects as go
import requests
import streamlit as st

logger = logging.getLogger(__name__)

import os
API_BASE_URL = os.environ.get("LITERAVORE_API_URL", "http://localhost:8000")


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
    paper_id = result.get("paper_id") or result.get("id", "")
    title = result.get("title", "Untitled")
    authors = format_authors(result.get("authors"))
    venue = result.get("venue") or result.get("conference") or "Unknown"
    score = result.get("score")
    abstract = result.get("abstract") or ""
    tags: list[str] = result.get("tags") or []
    openreview_url = result.get("openreview_url") or (
        f"https://openreview.net/forum?id={paper_id}" if paper_id else ""
    )

    summary = result.get("summary") or ""

    with st.container():
        col_main, col_action = st.columns([5, 1])
        with col_main:
            if openreview_url:
                st.markdown(
                    f"**<a href='{openreview_url}' target='_blank'>{title}</a>**",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"**{title}**")
            st.caption(f"Authors: {authors}  |  Venue: {venue}")
            if score is not None:
                st.caption(f"Score: {score:.4f}")
            if summary:
                st.markdown(summary)
            if tags:
                tag_str = "  ".join(f"`{t}`" for t in tags)
                st.markdown(tag_str)
        with col_action:
            if paper_id:
                if st.button("Details", key=f"detail_{index}_{paper_id}"):
                    st.session_state["selected_paper_id"] = paper_id
                    st.rerun()
        st.divider()


def render_paper_detail(paper_id: str) -> None:
    """Render the full paper detail view."""
    detail = get_paper_detail(paper_id)
    if not detail:
        st.warning("Could not load paper details. The API may be unavailable.")
        return

    title = detail.get("title", "Untitled")
    authors_list: list[str] = detail.get("authors") or []
    if isinstance(authors_list, list):
        authors_full = ", ".join(
            a.get("name", str(a)) if isinstance(a, dict) else str(a) for a in authors_list
        )
    else:
        authors_full = str(authors_list)
    venue = detail.get("venue") or detail.get("conference") or "Unknown"
    abstract = detail.get("abstract") or ""
    summary = detail.get("summary") or ""
    tags = detail.get("tags") or []
    openreview_url = detail.get("openreview_url") or detail.get("url") or ""
    pdf_url = detail.get("pdf_url") or ""

    # Top row: links (left) + back button (right)
    link_parts = []
    if openreview_url:
        link_parts.append(f"[View on OpenReview]({openreview_url})")
    if pdf_url:
        link_parts.append(f"[Download PDF]({pdf_url})")

    col_links, col_back = st.columns([5, 1])
    with col_links:
        if link_parts:
            st.markdown("  |  ".join(link_parts))
    with col_back:
        if st.button("← Results", use_container_width=True):
            st.session_state["selected_paper_id"] = None
            st.rerun()

    st.markdown(f"### {title}")
    st.caption(f"{authors_full}  |  {venue}")

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


def get_umap_coords() -> list[dict[str, Any]]:
    """Fetch UMAP 2D coordinates for all papers from the API, cached in session state."""
    if "umap_coords" not in st.session_state:
        data = make_api_request("/umap", timeout=60)
        st.session_state["umap_coords"] = data.get("points", []) if data else []
    return st.session_state["umap_coords"]  # type: ignore[no-any-return]


def build_scatter_plot(results: list[dict[str, Any]]) -> go.Figure:
    """Build a UMAP 2D scatter plot: all papers in grey, search results in red."""
    umap_points = get_umap_coords()
    if not umap_points:
        return go.Figure()

    result_ids = {r.get("paper_id") or r.get("id", "") for r in results}

    background = [p for p in umap_points if p["paper_id"] not in result_ids]
    highlighted = [p for p in umap_points if p["paper_id"] in result_ids]

    # Build rank lookup for hover text
    rank_lookup = {
        (r.get("paper_id") or r.get("id", "")): r.get("rank", "")
        for r in results
    }

    fig = go.Figure()

    if background:
        fig.add_trace(
            go.Scatter(
                x=[p["x"] for p in background],
                y=[p["y"] for p in background],
                mode="markers",
                marker=dict(size=7, color="rgba(180,180,180,0.5)"),
                name="All papers",
                text=[p["title"] for p in background],
                hovertemplate="<b>%{text}</b><extra></extra>",
            )
        )

    if highlighted:
        fig.add_trace(
            go.Scatter(
                x=[p["x"] for p in highlighted],
                y=[p["y"] for p in highlighted],
                mode="markers+text",
                marker=dict(size=11, color="red", line=dict(width=1, color="white")),
                name="Search results",
                text=[str(rank_lookup.get(p["paper_id"], "")) for p in highlighted],
                textposition="top center",
                textfont=dict(color="white", size=10),
                customdata=[p["title"] for p in highlighted],
                hovertemplate="<b>%{customdata}</b><br>Rank: %{text}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Paper Map (UMAP) — search results in red",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        height=450,
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
            f"The Literavore API is not reachable at {API_BASE_URL}. "
            "Start it with `literavore serve` to enable search."
        )
    else:
        paper_count = health.get("paper_count") or health.get("papers_count") or health.get("total_papers", "?")
        st.markdown(
            f"<span style='color:#00c853;'>API connected — {paper_count} papers indexed</span>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Search form
    with st.form("search_form"):
        col_input, col_button = st.columns([5, 1])
        with col_input:
            query = st.text_input(
                "Search query",
                placeholder="e.g. diffusion models for image generation",
                label_visibility="collapsed",
            )
        with col_button:
            search_clicked = st.form_submit_button("Search", type="primary", use_container_width=True)

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
