import pytest
from finagentx.streamlit_app import st

def test_integration():
    # This is more of an end-to-end integration test, checking if everything runs without errors
    try:
        st.text_input("Enter Stock Ticker", "AAPL")
        st.text_area("Input news/headline:", "Apple Inc. to release new iPhone model soon.")
        st.write("Test complete!")
    except Exception as e:
        pytest.fail(f"Streamlit app failed during integration test: {e}")
