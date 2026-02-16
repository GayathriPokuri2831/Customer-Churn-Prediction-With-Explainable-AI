# theme.py
import streamlit as st

def init_theme():
    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark"  # default

    theme = st.session_state["theme"]

    # Inject CSS for the current page
    st.markdown(
        f"""
        <style>
            :root {{
                --background: {"#050913" if theme == 'dark' else "#fae0a2"};
                --text: {'#e2e8f0' if theme == 'dark' else '#1e293b'};
                --primary: {"#d31313" if theme == 'dark' else "#4376e2"};
            }}
            [data-testid="stAppViewContainer"] {{
                background-color: var(--background);
                color: var(--text);
            }}
            .stButton > button {{
                background-color: var(--primary);
                color: white;
            }}
            /* Add more styles as needed */
        </style>
        """,
        unsafe_allow_html=True
    )

def toggle_theme():
        if st.toggle("🌙 Dark Mode", value=(st.session_state.theme=="dark")):
            st.session_state.theme = "dark"
        else:
            st.session_state.theme = "light"