"""
Consumer Electronics Sentiment Analyzer - Streamlit Chat Interface
Main entry point for the web application
"""

import os
import sys
import streamlit as st
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Diagnostic info for Streamlit Cloud debugging
if os.getenv("STREAMLIT_SHARING") or os.getenv("STREAMLIT_CLOUD"):
    print("=" * 60)
    print("STREAMLIT CLOUD DIAGNOSTICS")
    print(f"Python: {sys.version}")
    print(f"CWD: {os.getcwd()}")
    print(f"GROQ_API_KEY present: {bool(os.getenv('GROQ_API_KEY'))}")
    print(f"SUPABASE_URL present: {bool(os.getenv('SUPABASE_URL'))}")
    print(f"SUPABASE_SERVICE_KEY present: {bool(os.getenv('SUPABASE_SERVICE_KEY'))}")
    print("=" * 60)

# Import RAG pipeline
from rag.pipeline import RAGPipeline
from rag.query_classifier import get_example_questions
from supabase_db.db_client import get_client

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Consumer Electronics Sentiment Analyzer",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    /* Main chat container */
    .main {
        background-color: #0e1117;
    }

    /* Source cards */
    .source-card {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }

    .source-card-title {
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .source-card-meta {
        font-size: 0.85em;
        color: #9ca3af;
        margin-bottom: 8px;
    }

    .source-card-link {
        color: #4CAF50;
        text-decoration: none;
    }

    .source-card-link:hover {
        text-decoration: underline;
    }

    /* Stats badges */
    .stat-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: 500;
        margin-right: 8px;
    }

    .stat-positive {
        background-color: #22c55e;
        color: white;
    }

    .stat-negative {
        background-color: #ef4444;
        color: white;
    }

    .stat-neutral {
        background-color: #6b7280;
        color: white;
    }

    /* Example questions */
    .example-question {
        background-color: #1e2130;
        padding: 10px;
        border-radius: 6px;
        margin: 6px 0;
        cursor: pointer;
        border: 1px solid #374151;
        transition: all 0.2s;
    }

    .example-question:hover {
        background-color: #2d3748;
        border-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

def initialize_session_state():
    """Initialize all session state variables"""

    # Chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": """👋 **Welcome to the Consumer Electronics Sentiment Analyzer!**

I analyze 25,000+ Amazon product reviews to help you understand what people think about electronics.

**Try asking me:**
- "What do people think about iPhone 15?"
- "Are gaming laptops worth it?"
- "Should I buy MacBook or Dell laptop?"

Or type **"What can you do?"** to learn more!""",
                "sources": [],
                "metadata": {}
            }
        ]

    # RAG Pipeline (lazy loaded)
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
        st.session_state.pipeline_loaded = False

    # Filters
    if 'selected_subreddit' not in st.session_state:
        st.session_state.selected_subreddit = None

    if 'selected_sentiment' not in st.session_state:
        st.session_state.selected_sentiment = None

    if 'selected_days' not in st.session_state:
        st.session_state.selected_days = 365

    if 'response_style' not in st.session_state:
        st.session_state.response_style = "balanced"

    # Stats cache
    if 'db_stats' not in st.session_state:
        st.session_state.db_stats = None
        st.session_state.last_stats_update = None

initialize_session_state()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load and cache the RAG pipeline"""
    return RAGPipeline(verbose=False)


def get_database_stats() -> Dict[str, Any]:
    """Get database statistics (cached for 5 minutes)"""
    now = datetime.now()

    # Check cache
    if (st.session_state.db_stats is not None and
        st.session_state.last_stats_update is not None):
        time_since_update = (now - st.session_state.last_stats_update).seconds
        if time_since_update < 300:  # 5 minutes
            return st.session_state.db_stats

    # Fetch fresh stats
    try:
        client = get_client()
        stats = client.get_stats()

        st.session_state.db_stats = stats
        st.session_state.last_stats_update = now

        return stats
    except Exception as e:
        return {
            'error': str(e),
            'total_posts': 0,
            'posts_with_embeddings': 0
        }


def format_sentiment_badge(sentiment: str) -> str:
    """Format sentiment as colored badge"""
    sentiment_lower = sentiment.lower()
    if sentiment_lower == 'positive':
        return f'<span class="stat-badge stat-positive">✓ Positive</span>'
    elif sentiment_lower == 'negative':
        return f'<span class="stat-badge stat-negative">✗ Negative</span>'
    else:
        return f'<span class="stat-badge stat-neutral">○ Neutral</span>'


def display_source_card(post: Dict[str, Any], index: int):
    """Display a source post as a card"""
    source = post.get('source', 'amazon_reviews')
    subreddit = post.get('subreddit', 'unknown')
    title = post.get('title', 'No title')
    sentiment = post.get('sentiment_label', 'neutral')
    similarity = post.get('similarity', 0) * 100  # Convert to percentage
    permalink = post.get('permalink', '')
    url = post.get('url', '')
    score = post.get('score', 0)

    # Build Amazon URL (from url field)
    product_url = url if url else f"https://www.amazon.com/dp/{subreddit}"
    
    # Format the source display (Amazon reviews instead of Reddit)
    if source == 'amazon_reviews':
        source_display = f"Amazon ({subreddit[:8]}...)" if len(subreddit) > 8 else f"Amazon ({subreddit})"
    else:
        source_display = f"{source}"

    # Format sentiment badge
    sentiment_html = format_sentiment_badge(sentiment)

    # Display source card
    st.markdown(f"""
    <div class="source-card">
        <div class="source-card-title">{index}. {title}</div>
        <div class="source-card-meta">
            {source_display} • {score} stars • {similarity:.1f}% relevant • {sentiment_html}
        </div>
        <a href="{product_url}" target="_blank" class="source-card-link">View on Amazon →</a>
    </div>
    """, unsafe_allow_html=True)


def display_message(message: Dict[str, Any]):
    """Display a single message in the chat"""
    role = message["role"]
    content = message["content"]
    sources = message.get("sources", [])
    metadata = message.get("metadata", {})

    with st.chat_message(role):
        # Display the answer
        st.markdown(content)

        # Display sources if available
        if sources and len(sources) > 0:
            with st.expander(f"📚 View {len(sources)} Sources", expanded=False):
                for i, source in enumerate(sources, 1):
                    display_source_card(source, i)

        # Display metadata if in debug mode
        if metadata and st.session_state.get('debug_mode', False):
            with st.expander("🔍 Debug Info", expanded=False):
                st.json(metadata)


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():
    """Render the sidebar with filters and stats"""

    with st.sidebar:
        st.title("📱 Sentiment Analyzer")

        # Database Statistics
        st.header("📊 Database Stats")

        stats = get_database_stats()

        if 'error' not in stats:
            total_posts = stats.get('total_posts', 0)
            posts_with_embeddings = stats.get('posts_with_embeddings', 0)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Posts", f"{total_posts:,}")
            with col2:
                st.metric("With Embeddings", f"{posts_with_embeddings:,}")

            # Sentiment distribution
            if 'sentiment_distribution' in stats:
                sent_dist = stats['sentiment_distribution']
                st.write("**Sentiment Distribution:**")
                st.progress(sent_dist.get('positive', 0) / total_posts if total_posts > 0 else 0,
                           text=f"Positive: {sent_dist.get('positive', 0):,}")
                st.progress(sent_dist.get('neutral', 0) / total_posts if total_posts > 0 else 0,
                           text=f"Neutral: {sent_dist.get('neutral', 0):,}")
                st.progress(sent_dist.get('negative', 0) / total_posts if total_posts > 0 else 0,
                           text=f"Negative: {sent_dist.get('negative', 0):,}")
        else:
            st.error(f"Failed to load stats: {stats.get('error', 'Unknown error')}")

        st.divider()

        # Search Filters
        st.header("🔍 Search Filters")

        # Source filter (Amazon reviews instead of subreddits)
        st.info("📦 Data source: Amazon Product Reviews")

        # Sentiment filter
        sentiment_options = ["All Sentiments", "positive", "negative", "neutral"]
        selected_sentiment_display = st.selectbox(
            "Sentiment",
            options=sentiment_options,
            index=0
        )
        st.session_state.selected_sentiment = None if selected_sentiment_display == "All Sentiments" else selected_sentiment_display

        # Date range filter
        days_options = {
            "Last 30 days": 30,
            "Last 90 days": 90,
            "Last 6 months": 180,
            "Last year": 365,
            "All time": 10000
        }
        selected_days_display = st.selectbox(
            "Time Range",
            options=list(days_options.keys()),
            index=3  # Default to "Last year"
        )
        st.session_state.selected_days = days_options[selected_days_display]

        st.divider()

        # Response Settings
        st.header("⚙️ Response Settings")

        style_options = {
            "Concise": "concise",
            "Balanced": "balanced",
            "Detailed": "detailed"
        }
        selected_style_display = st.selectbox(
            "Response Style",
            options=list(style_options.keys()),
            index=1  # Default to Balanced
        )
        st.session_state.response_style = style_options[selected_style_display]

        st.divider()

        # Example Questions
        st.header("💡 Example Questions")

        examples = get_example_questions(5)
        for example in examples:
            if st.button(example, key=f"example_{example[:20]}", use_container_width=True):
                st.session_state.example_clicked = example

        st.divider()

        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = [st.session_state.messages[0]]  # Keep welcome message
            st.rerun()

        # Debug mode toggle
        st.session_state.debug_mode = st.checkbox("Debug Mode", value=False)


# ============================================================
# MAIN CHAT INTERFACE
# ============================================================

def main():
    """Main application"""

    # Render sidebar
    render_sidebar()

    # Main content area
    st.title("💬 Chat with Sentiment Analyzer")

    # Display chat history
    for message in st.session_state.messages:
        display_message(message)

    # Check if example question was clicked
    if 'example_clicked' in st.session_state:
        user_input = st.session_state.example_clicked
        del st.session_state.example_clicked
    else:
        # Chat input
        user_input = st.chat_input("Ask me about consumer electronics sentiment...")

    # Handle user input
    if user_input:
        # Display user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "sources": [],
            "metadata": {}
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Lazy load pipeline
                if not st.session_state.pipeline_loaded:
                    st.session_state.pipeline = load_pipeline()
                    st.session_state.pipeline_loaded = True

                # Query pipeline
                try:
                    # Build conversation history (last 6 turns = 3 Q&A pairs)
                    # Format: [{"role": "user/assistant", "content": "..."}, ...]
                    conversation_history = []
                    if len(st.session_state.messages) > 1:
                        # Get last 6 messages (excluding current user input)
                        # This gives us up to 3 Q&A pairs of context
                        recent_messages = st.session_state.messages[-7:-1]  # -7 to -1 gives us last 6
                        for msg in recent_messages:
                            conversation_history.append({
                                "role": msg["role"],
                                "content": msg["content"]
                            })

                    result = st.session_state.pipeline.query(
                        question=user_input,
                        subreddit_filter=st.session_state.selected_subreddit,
                        sentiment_filter=st.session_state.selected_sentiment,
                        days_ago=st.session_state.selected_days,
                        style=st.session_state.response_style,
                        enable_conversational=True,
                        conversation_history=conversation_history if conversation_history else None
                    )

                    # Display answer
                    st.markdown(result['answer'])

                    # Display sources
                    sources = result.get('sources', [])
                    if sources and len(sources) > 0:
                        with st.expander(f"📚 View {len(sources)} Sources", expanded=False):
                            for i, source in enumerate(sources, 1):
                                display_source_card(source, i)

                    # Display timing info if available
                    metadata = result.get('metadata', {})
                    if 'timing' in metadata:
                        timing = metadata['timing']
                        st.caption(f"⏱️ Response generated in {timing['total_time']:.2f}s")

                    # Save to message history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result['answer'],
                        "sources": sources,
                        "metadata": metadata
                    })

                except Exception as e:
                    error_message = f"❌ An error occurred: {str(e)}"
                    st.error(error_message)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                        "metadata": {"error": str(e)}
                    })


# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":
    main()
