"""Streamlit UI for the Pali Canon RAG Agent."""

import streamlit as st

from src.agent import PaliRAGAgent


# Page configuration
st.set_page_config(
    page_title="Pali Canon RAG",
    page_icon="📿",
    layout="wide",
)


def init_session_state():
    """Initialize session state variables."""
    if "agent" not in st.session_state:
        st.session_state.agent = PaliRAGAgent()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model" not in st.session_state:
        st.session_state.model = "ollama"


def render_sidebar():
    """Render the sidebar with settings."""
    with st.sidebar:
        st.title("Settings")

        # Model selection
        st.subheader("LLM Model")
        model_options = {
            "ollama": "Local (Ollama - Free)",
            "gemini": "Gemini 1.5 Flash",
            "claude": "Claude 3.5 Sonnet",
        }

        selected_model = st.selectbox(
            "Choose model:",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=list(model_options.keys()).index(st.session_state.model),
        )

        if selected_model != st.session_state.model:
            st.session_state.model = selected_model
            try:
                st.session_state.agent.set_model(selected_model)
                st.success(f"Switched to {model_options[selected_model]}")
            except ValueError as e:
                st.error(str(e))

        # Status
        st.divider()
        st.subheader("Status")

        if st.session_state.agent.is_ready():
            doc_count = st.session_state.agent.get_document_count()
            st.success(f"Ready - {doc_count:,} chunks indexed")
        else:
            st.warning("No suttas indexed yet")
            st.info("Run `python ingest.py` to index suttas")

        # Clear chat button
        st.divider()
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # About section
        st.divider()
        st.subheader("About")
        st.markdown("""
        **Pali Canon RAG Agent**

        Ask questions about the Early Buddhist texts
        from the Pali Canon. Answers include citations
        to specific sutta passages.

        Data source: [SuttaCentral](https://suttacentral.net)
        """)


def render_chat():
    """Render the chat interface."""
    st.title("Pali Canon RAG Agent 📿")
    st.caption("Ask questions about the Early Buddhist texts")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources in expander for assistant messages
            if message["role"] == "assistant" and "citations" in message:
                citations = message["citations"]
                if citations:
                    with st.expander(f"View Sources ({len(citations)})"):
                        for i, c in enumerate(citations, 1):
                            st.markdown(f"**{i}. {c['title']}** ({c['sutta_uid']}: {c['segment_range']})")
                            st.markdown(f"*Score: {c['score']:.3f}*")
                            st.text(c["text"][:500] + "..." if len(c["text"]) > 500 else c["text"])
                            st.divider()

    # Chat input
    if prompt := st.chat_input("Ask about the Pali Canon..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            if not st.session_state.agent.is_ready():
                response_text = "No suttas have been indexed yet. Please run `python ingest.py` first."
                st.markdown(response_text)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "citations": [],
                })
            else:
                with st.spinner("Searching suttas and generating response..."):
                    try:
                        result = st.session_state.agent.ask(prompt)
                        st.markdown(result["answer"])

                        # Show sources
                        citations = result["citations"]
                        if citations:
                            with st.expander(f"View Sources ({len(citations)})"):
                                for i, c in enumerate(citations, 1):
                                    st.markdown(f"**{i}. {c['title']}** ({c['sutta_uid']}: {c['segment_range']})")
                                    st.markdown(f"*Score: {c['score']:.3f}*")
                                    st.text(c["text"][:500] + "..." if len(c["text"]) > 500 else c["text"])
                                    st.divider()

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "citations": citations,
                        })

                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "citations": [],
                        })


def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
