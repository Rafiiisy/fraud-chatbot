"""
Response Display Component
Handles displaying responses with charts, metrics, and data
"""
import streamlit as st
import pandas as pd


class ResponseDisplay:
    def __init__(self):
        pass
    
    def render_response(self, response_data):
        """Render the response to the user's question"""
        if not response_data:
            return
        
        st.markdown(f'<div class="response-card">', unsafe_allow_html=True)
        
        # Response title
        st.markdown(f"### {response_data['title']}")
        
        # Confidence indicator
        if 'confidence' in response_data and response_data['confidence'] > 0:
            confidence = response_data['confidence']
            confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
            st.markdown(f"**Confidence:** :{confidence_color}[{confidence:.1%}]")
        
        # Explanation
        st.markdown(response_data['explanation'])
        
        # Metrics (if available) - skip for 'value_analysis' responses to avoid duplication with insights
        if 'metrics' in response_data and response_data['metrics'] and response_data.get('type') != 'value_analysis':
            st.subheader("📊 Key Metrics")
            cols = st.columns(len(response_data['metrics']))
            for i, (key, value) in enumerate(response_data['metrics'].items()):
                with cols[i]:
                    st.metric(
                        label=key.replace('_', ' ').title(),
                        value=value if isinstance(value, (int, float)) else f"{value:.1f}%" if 'rate' in key or 'share' in key else value
                    )
        
        # Chart (if available)
        if 'chart' in response_data and response_data['chart'] is not None:
            st.subheader("📈 Visualization")
            try:
                # Check if chart is a Plotly figure object or needs to be created
                if hasattr(response_data['chart'], 'show'):
                    # It's already a Plotly figure
                    st.plotly_chart(response_data['chart'], use_container_width=True)
                else:
                    # It's chart configuration data, create the chart
                    st.plotly_chart(response_data['chart'], use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying chart: {e}")
                st.write("Chart data available but could not be displayed.")
        
        # Content list (if available)
        if 'content' in response_data and response_data['content']:
            st.subheader("📋 Key Insights")
            for item in response_data['content']:
                st.markdown(f"• {item}")
        
        # Data table (if available)
        if 'data' in response_data and response_data['data'] is not None:
            st.subheader("📊 Data Table")
            if hasattr(response_data['data'], 'head'):  # pandas DataFrame
                st.dataframe(response_data['data'], use_container_width=True)
            else:
                st.write(response_data['data'])
        
        # SQL Query (if available and user wants to see it)
        if 'sql_query' in response_data and response_data['sql_query']:
            with st.expander("🔍 View SQL Query"):
                st.code(response_data['sql_query'], language='sql')
        
        # Sources (if available)
        if 'sources' in response_data and response_data['sources']:
            st.subheader("📚 Sources")
            for source in response_data['sources']:
                st.markdown(f"- {source}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_chat_history(self):
        """Render chat history"""
        if st.session_state.chat_history:
            st.header("💬 Chat History")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                with st.expander(f"Question {len(st.session_state.chat_history) - i}: {chat['question'][:50]}..."):
                    st.markdown(f"**Question:** {chat['question']}")
                    st.markdown(f"**Time:** {chat['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.markdown("**Response:**")
                    self.render_response(chat['response'])
