"""
ReadmitRisk - Enhanced Patient-Facing 30-Day Hospital Readmission Risk Calculator
A Streamlit web app with AI-powered health insights using Google Gemini
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from io import BytesIO

# Optional: Google Gemini AI Integration
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Optional: PDF Generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.pdfgen import canvas
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="ReadmitRisk - AI Health Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="auto"  # Auto-collapse on mobile
)

# Enhanced Custom CSS with Dark Mode Support & Glassmorphism
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Glassmorphism Cards */
    .info-card, .feature-card, .risk-container, .ai-insight, .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .feature-card:hover, .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.15);
        border-color: rgba(255, 255, 255, 0.2);
    }

    /* Headers */
    .main-header {
        font-size: 56px;
        background: linear-gradient(135deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 800;
        margin-bottom: 10px;
        letter-spacing: -1px;
        text-shadow: 0 10px 30px rgba(0, 201, 255, 0.3);
    }
    
    .sub-header {
        font-size: 20px;
        color: var(--text-color);
        opacity: 0.8;
        text-align: center;
        margin-bottom: 40px;
        font-weight: 300;
    }
    
    .section-header {
        font-size: 24px;
        font-weight: 700;
        color: var(--text-color);
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid rgba(255, 255, 255, 0.1);
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Risk Indicators */
    .risk-high {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.1) 0%, rgba(238, 90, 111, 0.1) 100%);
        border: 1px solid rgba(255, 107, 107, 0.3);
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, rgba(255, 217, 61, 0.1) 0%, rgba(255, 159, 0, 0.1) 100%);
        border: 1px solid rgba(255, 217, 61, 0.3);
    }
    
    .risk-low {
        background: linear-gradient(135deg, rgba(107, 207, 127, 0.1) 0%, rgba(46, 204, 113, 0.1) 100%);
        border: 1px solid rgba(107, 207, 127, 0.3);
    }
    
    .risk-label {
        font-size: 28px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 15px;
        color: var(--text-color);
    }
    
    .risk-description {
        font-size: 16px;
        line-height: 1.6;
        color: var(--text-color);
        opacity: 0.9;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: #000;
        font-size: 18px;
        font-weight: 700;
        padding: 15px 40px;
        border: none;
        border-radius: 12px;
        box-shadow: 0 10px 20px rgba(0, 201, 255, 0.2);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 15px 30px rgba(0, 201, 255, 0.4);
    }

    /* AI Insight Box */
    .ai-insight {
        border: 1px solid #00C9FF;
        background: rgba(0, 201, 255, 0.05);
    }
    
    .ai-badge {
        display: inline-block;
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: #000;
        padding: 5px 12px;
        border-radius: 8px;
        font-size: 11px;
        font-weight: 800;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Metrics */
    .metric-card {
        text-align: center;
        background: rgba(255, 255, 255, 0.03);
    }
    
    .metric-value {
        font-size: 36px;
        font-weight: 800;
        margin: 10px 0;
        background: linear-gradient(135deg, #FFF 0%, #CCC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.7;
        color: var(--text-color);
    }

    /* Feature List */
    .feature-card {
        border-left: 4px solid #00C9FF;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 5px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.3);
    }
    
    /* Mobile Responsive Styles */
    @media screen and (max-width: 768px) {
        .main-header {
            font-size: 36px;
        }
        
        .sub-header {
            font-size: 16px;
        }
        
        .section-header {
            font-size: 20px;
        }
        
        .info-card, .feature-card, .risk-container, .ai-insight, .metric-card {
            padding: 15px;
            margin: 10px 0;
        }
        
        .metric-value {
            font-size: 28px;
        }
        
        .risk-label {
            font-size: 22px;
        }
        
        .risk-description {
            font-size: 14px;
        }
        
        /* Sidebar mobile optimization */
        [data-testid="stSidebar"] {
            max-height: 100vh;
            overflow-y: auto;
        }
        
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            font-size: 13px;
        }
        
        [data-testid="stSidebar"] .info-card {
            padding: 10px;
            font-size: 12px;
        }
    }
    
    /* Sidebar styling for all devices */
    [data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-size: 14px;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Gemini AI (Robust Error Handling)
# Global variables for Gemini configuration
GEMINI_API_KEYS = []
GEMINI_CURRENT_KEY_INDEX = 0
GEMINI_MODEL_NAME = "gemini-2.5-flash"  # Default model - can be changed

def load_gemini_config():
    """Load Gemini API keys and model configuration"""
    global GEMINI_API_KEYS, GEMINI_MODEL_NAME
    api_keys = []
    
    try:
        # Try to load from secrets.toml
        if hasattr(st, 'secrets'):
            # Support single key (backward compatible)
            if "GEMINI_API_KEY" in st.secrets:
                key = st.secrets["GEMINI_API_KEY"]
                if key and key.strip():
                    api_keys.append(key.strip())
            
            # Support multiple keys
            if "GEMINI_API_KEYS" in st.secrets:
                keys = st.secrets["GEMINI_API_KEYS"]
                if isinstance(keys, list):
                    api_keys.extend([k.strip() for k in keys if k and k.strip()])
                elif isinstance(keys, str):
                    # Support comma-separated string
                    api_keys.extend([k.strip() for k in keys.split(',') if k.strip()])
            
            # Load model configuration
            if "GEMINI_MODEL" in st.secrets:
                GEMINI_MODEL_NAME = st.secrets["GEMINI_MODEL"]
    except (FileNotFoundError, KeyError, Exception) as e:
        print(f"Error loading secrets: {e}")
    
    # Fallback to environment variables
    if not api_keys:
        env_key = os.getenv("GEMINI_API_KEY")
        if env_key:
            api_keys.append(env_key)
        
        env_keys = os.getenv("GEMINI_API_KEYS")
        if env_keys:
            api_keys.extend([k.strip() for k in env_keys.split(',') if k.strip()])
    
    # Load model from environment
    env_model = os.getenv("GEMINI_MODEL")
    if env_model:
        GEMINI_MODEL_NAME = env_model
    
    # Remove duplicates while preserving order
    GEMINI_API_KEYS = list(dict.fromkeys(api_keys))
    
    # Debug logging
    if GEMINI_API_KEYS:
        print(f"✓ Loaded {len(GEMINI_API_KEYS)} Gemini API key(s)")
    else:
        print("✗ No Gemini API keys found in secrets or environment variables")
    
    return len(GEMINI_API_KEYS) > 0

def get_next_api_key():
    """Get next API key using round-robin rotation"""
    global GEMINI_CURRENT_KEY_INDEX
    if not GEMINI_API_KEYS:
        return None
    
    key = GEMINI_API_KEYS[GEMINI_CURRENT_KEY_INDEX]
    GEMINI_CURRENT_KEY_INDEX = (GEMINI_CURRENT_KEY_INDEX + 1) % len(GEMINI_API_KEYS)
    return key

def init_gemini():
    """Initialize Google Gemini AI with multi-key support and rotation"""
    if not GEMINI_AVAILABLE:
        print("✗ Google Generative AI package not available")
        return None
    
    if not load_gemini_config():
        print("✗ Failed to load Gemini API configuration")
        return None
    
    try:
        # Try to configure with first available key
        api_key = get_next_api_key()
        if api_key:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            print(f"✓ Gemini AI initialized successfully with model: {GEMINI_MODEL_NAME}")
            return model
        else:
            print("✗ No API key available for Gemini initialization")
    except Exception as e:
        print(f"✗ Gemini initialization error: {e}")
    
    return None

def get_gemini_response(prompt, max_retries=None):
    """Get Gemini response with automatic key rotation on failure"""
    if not gemini_model:
        return None
    
    # Default to number of available keys
    if max_retries is None:
        max_retries = len(GEMINI_API_KEYS) if GEMINI_API_KEYS else 1
    
    for attempt in range(max_retries):
        try:
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API attempt {attempt + 1} failed: {e}")
            
            # Try next API key if available
            if attempt < max_retries - 1 and GEMINI_API_KEYS:
                try:
                    next_key = get_next_api_key()
                    if next_key:
                        genai.configure(api_key=next_key)
                        print(f"Rotated to next API key (index: {GEMINI_CURRENT_KEY_INDEX})")
                except Exception as config_error:
                    print(f"Key rotation failed: {config_error}")
    
    return None

gemini_model = init_gemini()

# PDF Generation Function
def generate_pdf_report(risk_percentage, risk_category, patient_data, risk_factors, recommendations):
    """Generate a professional PDF health report"""
    print("\n=== PDF GENERATION STARTED ===")
    print(f"Risk: {risk_percentage}%, Category: {risk_category}")
    print(f"Patient Data: {patient_data}")
    print(f"Risk Factors: {risk_factors}")
    print(f"Recommendations count: {len(recommendations)}")
    
    if not PDF_AVAILABLE:
        print("ERROR: PDF_AVAILABLE is False")
        return None
    
    try:
        print("Creating BytesIO buffer...")
        buffer = BytesIO()
        print("Creating SimpleDocTemplate...")
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        story = []
        print("PDF setup complete, building content...")
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#00C9FF'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#00C9FF'),
            spaceAfter=10,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=11,
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        )
        
        # Title
        story.append(Paragraph("ReadmitRisk AI", title_style))
        story.append(Paragraph("30-Day Hospital Readmission Risk Report", styles['Heading3']))
        story.append(Spacer(1, 0.3*inch))
        
        # Report Info
        story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", body_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Risk Assessment Section
        story.append(Paragraph("Risk Assessment", heading_style))
        
        # Risk level color
        if risk_percentage > 50:
            risk_color = colors.HexColor('#ff6b6b')
            risk_bg = colors.HexColor('#ffebee')
        elif risk_percentage > 30:
            risk_color = colors.HexColor('#ff9f00')
            risk_bg = colors.HexColor('#fff3cd')
        else:
            risk_color = colors.HexColor('#2ecc71')
            risk_bg = colors.HexColor('#d4edda')
        
        # Risk table
        risk_data = [
            ['30-Day Readmission Risk', f"{risk_percentage:.1f}%"],
            ['Risk Level', risk_category]
        ]
        
        risk_table = Table(risk_data, colWidths=[3*inch, 3*inch])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), risk_bg),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.white)
        ]))
        
        story.append(risk_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Patient Information
        story.append(Paragraph("Patient Information", heading_style))
        patient_info = [
            ['Age Group', str(patient_data.get('age', 'N/A'))],
            ['Gender', str(patient_data.get('gender', 'N/A'))],
            ['Days in Hospital', str(patient_data.get('time_in_hospital', 'N/A'))],
            ['Prior Hospitalizations', str(patient_data.get('number_inpatient', 'N/A'))]
        ]
        
        patient_table = Table(patient_info, colWidths=[3*inch, 3*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.white)
        ]))
        
        story.append(patient_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Risk Factors Section
        story.append(Paragraph("Identified Risk Factors", heading_style))
        
        if risk_factors and len(risk_factors) > 0:
            for i, factor in enumerate(risk_factors, 1):
                story.append(Paragraph(f"{i}. {factor}", body_style))
        else:
            story.append(Paragraph("✓ No major risk factors identified. Keep following your care plan!", body_style))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Recommendations Section
        story.append(Paragraph("Personalized Recommendations", heading_style))
        
        for icon, title, desc in recommendations:
            # Clean the text to avoid XML issues
            clean_desc = str(desc).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(f"<b>{title}:</b> {clean_desc}", body_style))
        
        story.append(Spacer(1, 0.4*inch))
        
        # Disclaimer
        story.append(Paragraph("Medical Disclaimer", heading_style))
        disclaimer_text = """This report is generated for educational purposes only and should not replace 
        professional medical advice, diagnosis, or treatment. The risk assessment is based on statistical models 
        and may not account for all individual factors. Always consult with your healthcare provider for medical 
        decisions and follow their recommendations. If you experience any concerning symptoms, contact your 
        healthcare provider immediately."""
        
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=body_style,
            fontSize=9,
            textColor=colors.HexColor('#6c757d'),
            alignment=TA_JUSTIFY
        )
        
        story.append(Paragraph(disclaimer_text, disclaimer_style))
        
        # Footer
        story.append(Spacer(1, 0.3*inch))
        footer_style = ParagraphStyle(
            'Footer',
            parent=body_style,
            fontSize=9,
            textColor=colors.HexColor('#95a5a6'),
            alignment=TA_CENTER
        )
        story.append(Paragraph("ReadmitRisk AI © 2025 | INT234 Predictive Analytics Project", footer_style))
        story.append(Paragraph("Powered by Machine Learning & Google Gemini AI", footer_style))
        
        # Build PDF
        print(f"Building PDF with {len(story)} elements...")
        doc.build(story)
        print("PDF built successfully")
        
        pdf_bytes = buffer.getvalue()
        print(f"PDF bytes extracted: {len(pdf_bytes)} bytes")
        buffer.close()
        print("Buffer closed")
        
        print(f"=== PDF GENERATION COMPLETED: {len(pdf_bytes)} bytes ===")
        return pdf_bytes
        
    except Exception as e:
        print(f"\n!!! PDF GENERATION ERROR !!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        import traceback
        print("\nFull Traceback:")
        traceback.print_exc()
        print("=== PDF GENERATION FAILED ===")
        return None

# Load models (cached for performance)
@st.cache_resource
def load_models():
    """Load trained models and preprocessors"""
    try:
        model = joblib.load('models/best_readmission_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        return model, scaler, label_encoders, feature_names, None
    except FileNotFoundError as e:
        return None, None, None, None, str(e)

def get_ai_insights(risk_percentage, patient_data, risk_factors):
    """Generate personalized health insights using Gemini AI"""
    if not gemini_model:
        return None
    
    try:
        prompt = f"""You are a compassionate healthcare AI assistant helping a patient understand their hospital readmission risk.

Patient Risk Profile:
- Readmission Risk: {risk_percentage:.1f}%
- Risk Factors: {', '.join(risk_factors) if risk_factors else 'None identified'}

Key Patient Data:
- Age: {patient_data.get('age', 'Unknown')}
- Days in hospital: {patient_data.get('time_in_hospital', 'Unknown')}
- Prior hospitalizations: {patient_data.get('number_inpatient', 0)}
- Medications changed: {patient_data.get('change', 'Unknown')}

Provide a brief, empathetic, and actionable response (3-4 sentences) that:
1. Acknowledges their risk level
2. Highlights one positive aspect of their health profile
3. Offers one specific, practical recommendation to reduce readmission risk
4. Encourages them without causing alarm

Keep the tone supportive and patient-friendly. Do not use medical jargon."""

        return get_gemini_response(prompt)
    except Exception as e:
        return None

def create_risk_gauge(risk_percentage):
    """Create an interactive gauge chart for risk visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_percentage,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Readmission Risk", 'font': {'size': 24}},
        number = {'suffix': "%", 'font': {'size': 48}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 2},
            'bar': {'color': "#00C9FF"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "rgba(128,128,128,0.5)",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(107, 207, 127, 0.3)'},
                {'range': [30, 50], 'color': 'rgba(255, 217, 61, 0.3)'},
                {'range': [50, 100], 'color': 'rgba(255, 107, 107, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': risk_percentage
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': "Inter, sans-serif"}
    )
    
    return fig

def create_risk_factors_chart(risk_factors_data):
    """Create a horizontal bar chart for risk factors"""
    if not risk_factors_data:
        return None
    
    fig = go.Figure(go.Bar(
        x=list(risk_factors_data.values()),
        y=list(risk_factors_data.keys()),
        orientation='h',
        marker=dict(
            color='#00C9FF',
            line=dict(color='#00C9FF', width=1)
        ),
        text=list(risk_factors_data.values()),
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Key Risk Factors Impact",
        xaxis_title="Impact Level",
        yaxis_title="",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': "Inter, sans-serif"},
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )
    
    return fig

# Header
st.markdown("""
<div style="display: flex; align-items: center; justify-content: center; margin-bottom: 10px;">
    <svg width="60" height="60" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-right: 15px;">
        <path d="M19 3H5C3.89543 3 3 3.89543 3 5V19C3 20.1046 3.89543 21 5 21H19C20.1046 21 21 20.1046 21 19V5C21 3.89543 20.1046 3 19 3Z" fill="url(#paint0_linear)" stroke="url(#paint1_linear)" stroke-width="2"/>
        <path d="M12 8V16M8 12H16" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
        <defs>
            <linearGradient id="paint0_linear" x1="3" y1="3" x2="21" y2="21" gradientUnits="userSpaceOnUse">
                <stop stop-color="#00C9FF"/>
                <stop offset="1" stop-color="#92FE9D"/>
            </linearGradient>
            <linearGradient id="paint1_linear" x1="3" y1="3" x2="21" y2="21" gradientUnits="userSpaceOnUse">
                <stop stop-color="#00C9FF"/>
                <stop offset="1" stop-color="#92FE9D"/>
            </linearGradient>
        </defs>
    </svg>
    <div class="main-header" style="margin-bottom: 0;">ReadmitRisk AI</div>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="sub-header">Your Intelligent 30-Day Hospital Readmission Risk Calculator</div>', unsafe_allow_html=True)

# Check if models are available
model, scaler, label_encoders, feature_names, error = load_models()

if error:
    st.error("⚠️ **Models not found!** Please run the Jupyter notebook first to train and save the models.")
    st.info("""
    **Steps to get started:**
    1. Open `ReadmitRisk_Main_Notebook.ipynb`
    2. Run all cells to train models
    3. Models will be saved to the `models/` folder
    4. Restart this app
    """)
    st.stop()

# Success message with model info
col_status1, col_status2, col_status3 = st.columns([1, 2, 1])
with col_status2:
    st.success("✅ **AI Models Loaded** | Ready to Analyze")

# Introduction Card
with st.expander("ℹ️ About ReadmitRisk AI", expanded=False):
    st.markdown("""
<div class="info-card">
<h3>🤖 What is ReadmitRisk AI?</h3>
<p><strong>ReadmitRisk AI</strong> uses advanced machine learning to predict your chance of returning to the hospital 
within 30 days after discharge. Our AI has learned from over <strong>100,000 real patient records</strong>.</p>
<h4>✨ Features:</h4>
<ul>
<li>🎯 <strong>Accurate Predictions</strong> - Ensemble ML model (AUC: 0.68+)</li>
<li>🤖 <strong>AI Health Insights</strong> - Powered by Google Gemini (when enabled)</li>
<li>📊 <strong>Visual Risk Analysis</strong> - Interactive charts and metrics</li>
<li>💡 <strong>Personalized Recommendations</strong> - Tailored to your health profile</li>
</ul>
<h4>🔒 Privacy & Safety:</h4>
<p>Your data is processed securely and never stored. This is a <strong>screening tool</strong> for educational 
purposes and should not replace professional medical advice.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Create two-column layout for the form
st.markdown('<div class="section-header">📋 Patient Information</div>', unsafe_allow_html=True)

# Tab-based organization for better UX
tab1, tab2, tab3, tab4 = st.tabs(["👤 Personal & History", "🏥 Hospital Stay", "🩺 Medical Details", "💊 Medications"])

with tab1:
    col1, col2 = st.columns(2)

    
    with col1:
        st.markdown("##### Personal Information")
        age_options = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', 
                       '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
        age = st.selectbox("Age Group", age_options, index=5)
        gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
        race = st.selectbox("Race/Ethnicity", 
                           ["Caucasian", "African American", "Hispanic", "Asian", "Other"])
    
    with col2:
        st.markdown("##### Medical History")
        number_inpatient = st.number_input("Hospital Stays (Past Year)", 0, 20, 0,
                                          help="Number of times admitted to hospital in past 12 months")
        number_emergency = st.number_input("Emergency Visits (Past Year)", 0, 20, 0,
                                          help="Number of emergency room visits")
        number_outpatient = st.number_input("Outpatient Visits (Past Year)", 0, 40, 0,
                                           help="Number of outpatient clinic visits")

with tab2:
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("##### Hospital Stay Metrics")
        time_in_hospital = st.slider("Days in Hospital", 1, 14, 5)
        num_lab_procedures = st.number_input("Number of Lab Tests", 0, 150, 40)
        num_procedures = st.number_input("Non-Lab Procedures", 0, 10, 2)
        num_medications = st.number_input("Daily Medications", 1, 80, 15)
    
    with col4:
        st.markdown("##### Admission Details")
        admission_type = st.selectbox("Admission Type",
                                     ["Emergency", "Urgent", "Elective", "Newborn", "Other"])
        discharge_disposition = st.selectbox("Discharged To",
                                            ["Home", "Skilled Nursing Facility", "Another Hospital",
                                             "Home with Health Service", "Other"])
        admission_source = st.selectbox("Admitted From",
                                       ["Emergency Room", "Physician Referral", "Transfer",
                                        "Clinic Referral", "Other"])

with tab3:
    col5, col6 = st.columns(2)
    
    with col5:
        st.markdown("##### Diagnosis Information")
        primary_diagnosis = st.selectbox("Primary Diagnosis",
                                        ["Diabetes", "Circulatory (Heart)", "Respiratory (Lung)",
                                         "Digestive", "Injury", "Other"])
        number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 7)
    
    with col6:
        st.markdown("##### Lab Results")
        max_glu_serum = st.selectbox("Max Glucose Serum Test",
                                    ["None (Not Tested)", "Normal", ">200", ">300"])
        a1c_result = st.selectbox("HbA1c Test Result",
                                 ["None (Not Tested)", "Normal", ">7%", ">8%"])

with tab4:
    st.markdown("##### Diabetes Medications")
    col7, col8, col9, col10 = st.columns(4)
    
    with col7:
        metformin = st.selectbox("Metformin", ["No", "Steady", "Up", "Down"])
        insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
    
    with col8:
        glyburide = st.selectbox("Glyburide", ["No", "Steady", "Up", "Down"])
        glipizide = st.selectbox("Glipizide", ["No", "Steady", "Up", "Down"])
    
    with col9:
        glimepiride = st.selectbox("Glimepiride", ["No", "Steady", "Up", "Down"])
    
    with col10:
        diabetesMed = st.radio("Any Diabetes Med?", ["No", "Yes"])
        change = st.radio("Medication Changed?", ["No", "Yes"])

st.markdown("---")

# Calculate Risk Button
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    calculate_button = st.button("🔍 Analyze My Risk with AI", type="primary", use_container_width=True)

if calculate_button:
    
    with st.spinner("🤖 AI is analyzing your health data..."):
        
        # Map user inputs to encoded values (simplified encoding)
        # In production, use the actual label_encoders from training
        
        input_dict = {
            'age': age_options.index(age),
            'gender': 0 if gender == "Female" else 1,
            'race': {'Caucasian': 0, 'African American': 1, 'Hispanic': 2, 'Asian': 3, 'Other': 4}[race],
            'time_in_hospital': time_in_hospital,
            'num_lab_procedures': num_lab_procedures,
            'num_procedures': num_procedures,
            'num_medications': num_medications,
            'number_outpatient': number_outpatient,
            'number_emergency': number_emergency,
            'number_inpatient': number_inpatient,
            'number_diagnoses': number_diagnoses,
            'admission_type_id': {'Emergency': 1, 'Urgent': 2, 'Elective': 3, 'Newborn': 4, 'Other': 5}[admission_type],
            'discharge_disposition_id': {'Home': 1, 'Skilled Nursing Facility': 3, 'Another Hospital': 2, 
                                        'Home with Health Service': 6, 'Other': 18}[discharge_disposition],
            'admission_source_id': {'Emergency Room': 7, 'Physician Referral': 1, 'Transfer': 4,
                                   'Clinic Referral': 2, 'Other': 9}[admission_source],
            'diag_1': {'Diabetes': 250, 'Circulatory (Heart)': 410, 'Respiratory (Lung)': 486,
                      'Digestive': 535, 'Injury': 800, 'Other': 999}[primary_diagnosis],
            'diag_2': 250,  # Simplified - use same as primary
            'diag_3': 250,  # Simplified - use same as primary
            'max_glu_serum': {'None (Not Tested)': 0, 'Normal': 1, '>200': 2, '>300': 3}[max_glu_serum],
            'A1Cresult': {'None (Not Tested)': 0, 'Normal': 1, '>7%': 2, '>8%': 3}[a1c_result],
            'metformin': {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3}[metformin],
            'insulin': {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3}[insulin],
            'glyburide': {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3}[glyburide],
            'glipizide': {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3}[glipizide],
            'glimepiride': {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3}[glimepiride],
            'change': 0 if change == "No" else 1,
            'diabetesMed': 0 if diabetesMed == "No" else 1
        }
        
        # Create DataFrame matching training features
        # Fill missing features with defaults (0 or median values)
        input_df = pd.DataFrame([input_dict])
        
        # Ensure all training features are present
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0  # Default value for missing features
        
        # Reorder columns to match training
        input_df = input_df[feature_names]
        
        # Scale features (suppress warnings)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            input_scaled = scaler.transform(input_df)
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]  # Probability of readmission
        
        risk_percentage = probability * 100
        
    # Display Results
    st.markdown("---")
    st.markdown('<div class="section-header">📊 Your Results</div>', unsafe_allow_html=True)
    
    # Create three columns for metrics and gauge
    col_left, col_center, col_right = st.columns([1, 2, 1])
    
    with col_center:
        # Interactive gauge chart
        fig_gauge = create_risk_gauge(risk_percentage)
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
    
    with col_left:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Risk Score</div>
            <div class="metric-value">{risk_percentage:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_right:
        risk_category = "HIGH" if risk_percentage > 50 else ("MODERATE" if risk_percentage > 30 else "LOW")
        risk_class_card = "risk-high" if risk_percentage > 50 else ("risk-moderate" if risk_percentage > 30 else "risk-low")
        st.markdown(f"""
        <div class="metric-card {risk_class_card}">
            <div class="metric-label">Risk Level</div>
            <div class="metric-value" style="font-size: 28px;">{risk_category}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk interpretation with enhanced styling
    st.markdown("")  # Spacing
    
    if risk_percentage > 50:
        risk_class = "risk-high"
        risk_icon = "⚠️"
        risk_title = "HIGH RISK"
        risk_message = f"Your risk of hospital readmission within 30 days is <strong>{risk_percentage:.1f}%</strong>, which is considered high. This means approximately {int(risk_percentage)} out of 100 patients with a similar profile are readmitted."
        action_text = "🏥 <strong>Recommended Action:</strong> Please schedule a follow-up with your doctor within 3-5 days. Contact your healthcare provider immediately if symptoms worsen."
    elif risk_percentage > 30:
        risk_class = "risk-moderate"
        risk_icon = "🟡"
        risk_title = "MODERATE RISK"
        risk_message = f"Your risk of hospital readmission within 30 days is <strong>{risk_percentage:.1f}%</strong>, which is considered moderate. With proper care, you can reduce this risk significantly."
        action_text = "📅 <strong>Recommended Action:</strong> Schedule a follow-up appointment within 1-2 weeks and carefully follow all discharge instructions."
    else:
        risk_class = "risk-low"
        risk_icon = "✅"
        risk_title = "LOW RISK"
        risk_message = f"Your risk of hospital readmission within 30 days is <strong>{risk_percentage:.1f}%</strong>, which is considered low. You're on the right track!"
        action_text = "👍 <strong>Recommended Action:</strong> Continue following your discharge instructions and maintain your current care plan."
    
    st.markdown(f"""
    <div class="risk-container {risk_class}">
        <div class="risk-label">{risk_icon} {risk_title}</div>
        <div class="risk-description">{risk_message}</div>
        <hr style="margin: 20px 0; opacity: 0.3;">
        <div class="risk-description">{action_text}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Identify and display risk factors
    st.markdown('<div class="section-header">🔍 Key Risk Factors</div>', unsafe_allow_html=True)
    
    risk_factors = []
    risk_factors_impact = {}
    
    if number_inpatient > 2:
        risk_factors.append(f"Multiple hospital stays in past year ({number_inpatient} admissions)")
        risk_factors_impact["Prior Hospitalizations"] = min(number_inpatient * 2, 10)
    
    if insulin in ["Up", "Down"]:
        risk_factors.append(f"Insulin dose was changed ({insulin.lower()})")
        risk_factors_impact["Insulin Changes"] = 8
    
    if a1c_result in [">7%", ">8%"]:
        risk_factors.append(f"Elevated A1C levels ({a1c_result})")
        risk_factors_impact["High A1C"] = 7
    
    if number_emergency > 1:
        risk_factors.append(f"Recent emergency visits ({number_emergency} visits)")
        risk_factors_impact["Emergency Visits"] = min(number_emergency * 3, 10)
    
    if time_in_hospital > 7:
        risk_factors.append(f"Extended hospital stay ({time_in_hospital} days)")
        risk_factors_impact["Long Hospital Stay"] = min(time_in_hospital, 10)
    
    if change == "Yes":
        risk_factors.append("Diabetes medications were changed during admission")
        risk_factors_impact["Medication Changes"] = 6
    
    if num_medications > 20:
        risk_factors.append(f"High number of medications ({num_medications} daily)")
        risk_factors_impact["Polypharmacy"] = min(num_medications // 4, 10)
    
    col_factors1, col_factors2 = st.columns([1, 1])
    
    with col_factors1:
        if len(risk_factors) > 0:
            st.markdown("**Identified Risk Factors:**")
            for i, factor in enumerate(risk_factors[:6], 1):
                st.markdown(f"""
                <div class="feature-card">
                    <strong>{i}.</strong> {factor}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="feature-card" style="border-left-color: #6bcf7f;">
                ✅ <strong>No major risk factors identified.</strong> Keep following your care plan!
            </div>
            """, unsafe_allow_html=True)
    
    with col_factors2:
        if risk_factors_impact:
            fig_factors = create_risk_factors_chart(risk_factors_impact)
            st.plotly_chart(fig_factors, use_container_width=True, config={'displayModeBar': False})
    
    # AI-Powered Insights
    if gemini_model:
        st.markdown('<div class="section-header">🤖 AI Health Insights</div>', unsafe_allow_html=True)
        
        with st.spinner("Generating personalized insights..."):
            patient_summary = {
                'age': age,
                'time_in_hospital': time_in_hospital,
                'number_inpatient': number_inpatient,
                'change': change
            }
            ai_insight = get_ai_insights(risk_percentage, patient_summary, risk_factors)
        
        if ai_insight:
            st.markdown(f"""
            <div class="ai-insight">
                <div class="ai-badge">✨ POWERED BY GOOGLE GEMINI AI</div>
                <p style="font-size: 16px; line-height: 1.8; margin: 15px 0;">
                    {ai_insight}
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ **AI Insights Unavailable:** Gemini API keys not configured. Contact your administrator to enable AI-powered insights.")
    
    # Recommendations
    st.markdown('<div class="section-header">💡 Personalized Recommendations</div>', unsafe_allow_html=True)
    
    col_rec1, col_rec2, col_rec3 = st.columns(3)
    
    recommendations = [
        ("💊", "Medications", "Take all medications exactly as prescribed at the same time each day"),
        ("🩺", "Monitoring", "Monitor your blood sugar levels regularly and keep a log"),
        ("📅", "Follow-ups", "Keep all follow-up appointments with your healthcare team"),
        ("🚨", "Warning Signs", "Watch for fever, increased pain, or shortness of breath"),
        ("🥗", "Nutrition", "Maintain a healthy diet as advised by your care team"),
        ("📞", "Support", "Keep emergency contact numbers readily available")
    ]
    
    for i, (icon, title, desc) in enumerate(recommendations):
        col = [col_rec1, col_rec2, col_rec3][i % 3]
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <h4>{icon} {title}</h4>
                <p style="font-size: 14px; opacity: 0.8; margin: 8px 0 0 0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Download Report Button
    st.markdown("")
    col_download1, col_download2, col_download3 = st.columns([1, 1, 1])
    with col_download2:
        # Prepare patient data for PDF
        patient_data_for_pdf = {
            'age': age,
            'gender': gender,
            'time_in_hospital': time_in_hospital,
            'number_inpatient': number_inpatient,
            'change': change
        }
        
        # Text report
        report_data = f"""
READMITRISK AI - HEALTH REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

RISK ASSESSMENT:
- 30-Day Readmission Risk: {risk_percentage:.1f}%
- Risk Level: {risk_category}

RISK FACTORS:
{chr(10).join(f'- {factor}' for factor in risk_factors) if risk_factors else '- No major risk factors identified'}

RECOMMENDATIONS:
{chr(10).join(f'- {title}: {desc}' for _, title, desc in recommendations)}

DISCLAIMER:
This report is for educational purposes only and should not replace professional medical advice.
Always consult with your healthcare provider for medical decisions.
        """
        
        # Create two columns for download buttons
        col_txt, col_pdf = st.columns(2)
        
        with col_txt:
            st.download_button(
                label="📄 Download TXT",
                data=report_data,
                file_name=f"readmitrisk_report_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col_pdf:
            if PDF_AVAILABLE:
                try:
                    print("\n>>> Calling generate_pdf_report from download button...")
                    pdf_data = generate_pdf_report(
                        risk_percentage, 
                        risk_category, 
                        patient_data_for_pdf, 
                        risk_factors, 
                        recommendations
                    )
                    print(f">>> PDF data returned: {type(pdf_data)}, Length: {len(pdf_data) if pdf_data else 0}")
                    
                    if pdf_data and len(pdf_data) > 0:
                        print(f">>> Creating download button with {len(pdf_data)} bytes")
                        st.download_button(
                            label="📥 Download PDF",
                            data=pdf_data,
                            file_name=f"readmitrisk_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key=f"pdf_download_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        )
                        print(">>> Download button created successfully")
                    else:
                        print(">>> ERROR: PDF data is None or empty!")
                        st.error("❌ PDF generation failed - no data generated")
                except Exception as e:
                    print(f"\n>>> EXCEPTION in download button: {str(e)}")
                    st.error(f"❌ PDF Error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    st.code(traceback.format_exc())
            else:
                st.warning("Install reportlab for PDF: `pip install reportlab`")

# Sidebar
with st.sidebar:
    st.markdown("### 📖 Model Information")
    st.markdown("""
<div class="info-card" style="padding: 15px;">
<p><strong>🤖 Model Type:</strong><br>Voting Ensemble</p>
<p style="font-size: 12px; opacity: 0.8;">
• Random Forest<br>
• XGBoost<br>
• Logistic Regression
</p>
<p><strong>📊 Training Data:</strong><br>101,767 patients</p>
<p><strong>🎯 Performance:</strong><br>AUC-ROC: 0.6884<br>Accuracy: ~68%</p>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### ⚕️ Medical Disclaimer")
    st.warning("""
    This tool is for **educational purposes only**. 
    
    It should not replace professional medical advice, diagnosis, or treatment.
    
    Always consult your healthcare provider.
    """)
    
    st.markdown("---")
    
    st.markdown("### 🔧 Settings")
    
    if GEMINI_AVAILABLE:
        if gemini_model:
            st.success("✅ AI Insights: Enabled")
        else:
            st.info("🔑 AI Insights: Not configured")
    else:
        st.info("💡 AI Insights: Unavailable")
    
    st.markdown("---")
    
    st.markdown("### 👨‍💻 About")
    st.markdown("""
    **Developer:** Ashutosh Kumar Singh
    
    **Dataset:** UCI Diabetes 130-US Hospitals
    
    **Technologies:**
    - Python, Scikit-learn
    - XGBoost, Streamlit
    - Plotly, AI-Powered Insights
    
    © 2025 ReadmitRisk Project
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #95a5a6; padding: 30px 20px;">
<p style="font-size: 16px; margin-bottom: 10px;">
<strong>ReadmitRisk AI</strong> © 2025 | Developed by Ashutosh Kumar Singh
</p>
<p style="font-size: 13px;">
Built with ❤️ using Streamlit • Powered by Machine Learning & AI
</p>
<p style="font-size: 11px; margin-top: 15px; color: #bdc3c7;">
This is an educational tool. Always seek professional medical advice for health decisions.
</p>
</div>
""", unsafe_allow_html=True)
