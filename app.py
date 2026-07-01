# =============================================================================
#  NutriWise AI – Personalized Nutrition Coach
#  Powered by Google Gemini (gemini-2.5-flash)
#
#  Multi-Agent Architecture:
#   Agent 1 – Nutrition Knowledge Agent
#   Agent 2 – Diet Planner Agent
#   Agent 3 – Health Advisory Agent
#   Agent 4 – Meal Analysis Agent
#
#  Run:
#    pip install flask google-genai
#    set GOOGLE_API_KEY=<your-google-ai-studio-key>
#    python app.py
# =============================================================================

import os
from flask import Flask, request, jsonify, render_template_string

# ---------------------------------------------------------------------------
# Google Gemini SDK import  (google-genai package)
# ---------------------------------------------------------------------------
from google import genai
from google.genai import types

app = Flask(__name__)

# =============================================================================
# Google Gemini Configuration
# Credentials are read from environment variables for security.
# Get your free API key at: https://aistudio.google.com/app/apikey
# =============================================================================
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyAByyTBbCoZ_Gh90uCGhGjjD-3PwlH7snI")

# Google Gemini model identifier
GEMINI_MODEL_ID = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Core AI function – all agents route through this single function.
# This is the central Google Gemini API call point.
# ---------------------------------------------------------------------------
def generate_response(prompt: str, max_tokens: int = 1024) -> str:
    """
    Send a prompt to Google Gemini and return the response.
    This function is the single integration point for all four agents.
    """
    try:
        # Initialise the Google Gemini client with the API key
        client = genai.Client(api_key=GOOGLE_API_KEY)

        # Call Google Gemini – this is where the model is invoked
        response = client.models.generate_content(
            model=GEMINI_MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
            )
        )

        # Extract text safely (gemini-2.5-flash is a thinking model)
        text = response.text
        if not text and response.candidates:
            parts = response.candidates[0].content.parts
            text = "".join(p.text for p in parts if hasattr(p, "text") and p.text)
        return text.strip() if text else "I could not generate a response. Please try again."

    except Exception as e:
        # Return a friendly error so the UI always shows something meaningful
        return (
            f"⚠️ Google Gemini error: {str(e)}\n\n"
            "Please verify your GOOGLE_API_KEY environment variable."
        )


# =============================================================================
# AGENT 1 – Nutrition Knowledge Agent
# Answers general nutrition questions using Google Gemini.
# =============================================================================
def nutrition_knowledge_agent(question: str) -> str:
    """
    Agent 1: Responds to general nutrition and food knowledge questions.
    Powered by Google Gemini.
    """
    prompt = f"""You are NutriWise, an expert nutrition coach powered by Google Gemini AI.
Answer the following nutrition question in a clear, informative, and educational way.
Structure your answer with:
- A direct answer
- Key nutritional facts
- Practical tips or recommendations
- Any important cautions if relevant

Question: {question}

Provide a helpful, accurate, and easy-to-understand response."""

    # Google Gemini call via the shared generate_response function
    return generate_response(prompt, max_tokens=800)


# =============================================================================
# AGENT 2 – Diet Planner Agent
# Generates a personalized meal plan based on user profile.
# =============================================================================
def diet_planner_agent(age: int, gender: str, height: float, weight: float,
                       dietary_pref: str, activity_level: str, goal: str) -> str:
    """
    Agent 2: Creates a personalised daily meal plan.
    Uses Google Gemini to generate context-aware nutrition plans.
    """
    prompt = f"""You are NutriWise, an expert dietitian powered by Google Gemini AI.
Create a detailed personalized meal plan for the following person:

Profile:
- Age: {age} years
- Gender: {gender}
- Height: {height} cm
- Weight: {weight} kg
- Dietary Preference: {dietary_pref}
- Activity Level: {activity_level}
- Fitness Goal: {goal}

Provide a complete daily meal plan with:

1. DAILY TARGETS:
   - Daily Calorie Target (kcal)
   - Protein Recommendation (grams)
   - Carbohydrate Recommendation (grams)
   - Fat Recommendation (grams)

2. MEAL PLAN:
   Breakfast: (with portions)
   Mid-Morning Snack: (with portions)
   Lunch: (with portions)
   Evening Snack: (with portions)
   Dinner: (with portions)

3. HYDRATION & TIPS:
   - Water intake recommendation
   - 2-3 key tips for the goal

Make the plan realistic, balanced, and tailored to the {dietary_pref} dietary preference."""

    # Google Gemini call via the shared generate_response function
    return generate_response(prompt, max_tokens=1200)


# =============================================================================
# AGENT 3 – Health Advisory Agent
# Provides dietary advice for specific health conditions.
# =============================================================================
def health_advisory_agent(conditions: list) -> str:
    """
    Agent 3: Gives food and lifestyle recommendations for health conditions.
    Powered by Google Gemini.

    Always appends the medical disclaimer as required.
    """
    conditions_str = ", ".join(conditions) if conditions else "General Wellness"

    prompt = f"""You are NutriWise, a certified health and nutrition advisor powered by Google Gemini AI.
Provide comprehensive dietary and lifestyle recommendations for someone with the following condition(s): {conditions_str}

Structure your response as:

1. FOODS TO INCLUDE:
   List 8-10 beneficial foods with brief reasons

2. FOODS TO AVOID:
   List 8-10 foods to avoid with brief reasons

3. HEALTHY HABITS:
   List 5-6 daily habits that support managing this condition

4. LIFESTYLE RECOMMENDATIONS:
   List 4-5 lifestyle changes (exercise, sleep, stress management, etc.)

5. NUTRITIONAL PRIORITIES:
   Key nutrients to focus on for this condition

Be specific, practical, and evidence-based."""

    # Google Gemini call via the shared generate_response function
    response = generate_response(prompt, max_tokens=1200)

    # Always append the mandatory medical disclaimer
    disclaimer = (
        "\n\n---\n"
        "⚕️ **DISCLAIMER:** This information is for educational purposes only. "
        "It does not constitute medical advice, diagnosis, or treatment. "
        "Always consult a qualified healthcare professional or registered dietitian "
        "before making changes to your diet, especially if you have a medical condition."
    )
    return response + disclaimer


# =============================================================================
# AGENT 4 – Meal Analysis Agent
# Analyses a free-text meal description and provides nutritional feedback.
# =============================================================================
def meal_analysis_agent(meal_description: str) -> str:
    """
    Agent 4: Analyses a user's meal log and provides personalised feedback.
    Google Gemini evaluates nutritional quality and suggests improvements.
    """
    prompt = f"""You are NutriWise, an expert meal analyst powered by Google Gemini AI.
Analyse the following meal description and provide detailed nutritional feedback:

MEAL DESCRIPTION:
{meal_description}

Provide a structured analysis with:

1. NUTRITIONAL OVERVIEW:
   - Estimated caloric range
   - Macronutrient balance assessment (protein/carbs/fats)
   - Micronutrient highlights

2. NUTRITIONAL STRENGTHS:
   List 3-4 positive aspects of this meal plan

3. NUTRITIONAL GAPS & DEFICIENCIES:
   List 3-4 areas that need improvement

4. HEALTHIER ALTERNATIVES:
   Suggest specific swaps for less healthy items

5. IMPROVEMENT RECOMMENDATIONS:
   Provide 4-5 actionable tips to make these meals more nutritious

6. OVERALL SCORE:
   Give a nutrition score out of 10 with a brief explanation

Be constructive, specific, and encouraging in your feedback."""

    # Google Gemini call via the shared generate_response function
    return generate_response(prompt, max_tokens=1200)


# =============================================================================
# AGENT ORCHESTRATOR
# Routes incoming requests to the appropriate specialised agent.
# =============================================================================
def orchestrator(agent_type: str, payload: dict) -> str:
    """
    Orchestrator: Determines which agent handles the request and dispatches it.
    
    Supported agent_type values:
      - "nutrition_knowledge"
      - "diet_planner"
      - "health_advisory"
      - "meal_analysis"
    """
    if agent_type == "nutrition_knowledge":
        return nutrition_knowledge_agent(
            question=payload.get("question", "")
        )

    elif agent_type == "diet_planner":
        return diet_planner_agent(
            age=payload.get("age", 25),
            gender=payload.get("gender", "Not specified"),
            height=payload.get("height", 170),
            weight=payload.get("weight", 70),
            dietary_pref=payload.get("dietary_pref", "Omnivore"),
            activity_level=payload.get("activity_level", "Moderate"),
            goal=payload.get("goal", "General Wellness")
        )

    elif agent_type == "health_advisory":
        return health_advisory_agent(
            conditions=payload.get("conditions", ["General Wellness"])
        )

    elif agent_type == "meal_analysis":
        return meal_analysis_agent(
            meal_description=payload.get("meal_description", "")
        )

    else:
        return "❌ Unknown agent type. Please select a valid agent."


# =============================================================================
# HTML TEMPLATE – Complete Bootstrap 5 UI (embedded via render_template_string)
# =============================================================================
BASE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>NutriWise AI – Personalized Nutrition Coach</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"/>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet"/>
  <style>
    :root {
      --primary:   #1a7a4a;
      --primary-d: #145c38;
      --accent:    #f0a500;
      --sidebar-w: 260px;
      --bg:        #f4f7f5;
      --card-bg:   #ffffff;
      --text:      #1e2d24;
      --muted:     #6b7c72;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; font-family: 'Segoe UI', system-ui, sans-serif;
      background: var(--bg); color: var(--text);
      display: flex; min-height: 100vh;
    }
    /* ── Sidebar ── */
    #sidebar {
      width: var(--sidebar-w); background: #0f2d1e;
      display: flex; flex-direction: column;
      position: fixed; top: 0; left: 0; height: 100vh;
      z-index: 1000; transition: transform .3s ease;
    }
    #sidebar .brand {
      padding: 24px 20px 16px;
      border-bottom: 1px solid rgba(255,255,255,.1);
    }
    #sidebar .brand h5 {
      color: #fff; font-weight: 700; font-size: 1.05rem; margin: 0;
    }
    #sidebar .brand small { color: #8ab99e; font-size: .72rem; }
    #sidebar nav { flex: 1; padding: 16px 0; overflow-y: auto; }
    #sidebar nav .nav-label {
      color: #5a8a6e; font-size: .68rem; font-weight: 700;
      letter-spacing: .08em; text-transform: uppercase;
      padding: 10px 20px 4px;
    }
    #sidebar nav a {
      display: flex; align-items: center; gap: 12px;
      color: #b0cdc0; text-decoration: none;
      padding: 10px 20px; font-size: .88rem; border-radius: 0;
      transition: background .2s, color .2s;
    }
    #sidebar nav a:hover,
    #sidebar nav a.active {
      background: rgba(255,255,255,.08); color: #fff;
    }
    #sidebar nav a.active { border-left: 3px solid var(--accent); }
    #sidebar nav a i { font-size: 1.05rem; width: 20px; }
    #sidebar .sidebar-footer {
      padding: 16px 20px;
      border-top: 1px solid rgba(255,255,255,.1);
      color: #5a8a6e; font-size: .72rem; line-height: 1.5;
    }
    /* ── Main content ── */
    #main {
      margin-left: var(--sidebar-w);
      flex: 1; display: flex; flex-direction: column;
      min-height: 100vh;
    }
    /* ── Topbar ── */
    #topbar {
      background: #fff; border-bottom: 1px solid #dee8e3;
      padding: 14px 28px;
      display: flex; align-items: center; justify-content: space-between;
      position: sticky; top: 0; z-index: 900;
    }
    #topbar .page-title { font-size: 1.1rem; font-weight: 600; color: var(--text); margin: 0; }
    #topbar .badge-ibm {
      background: #052e78; color: #fff;
      font-size: .7rem; padding: 4px 10px; border-radius: 20px; font-weight: 600;
    }
    /* ── Content area ── */
    .content-area { padding: 28px; flex: 1; }
    /* ── Cards ── */
    .card { border: 1px solid #dde7e2; border-radius: 14px; background: var(--card-bg); box-shadow: 0 1px 4px rgba(0,0,0,.05); }
    .card-header-custom {
      background: linear-gradient(135deg, var(--primary), var(--primary-d));
      color: #fff; padding: 16px 20px; border-radius: 14px 14px 0 0;
      display: flex; align-items: center; gap: 10px;
    }
    .card-header-custom h5 { margin: 0; font-size: .98rem; font-weight: 600; }
    .card-header-custom .agent-badge {
      background: rgba(255,255,255,.2);
      font-size: .7rem; padding: 2px 8px; border-radius: 20px; margin-left: auto;
    }
    /* ── Agent overview cards ── */
    .agent-card {
      border-radius: 14px; padding: 22px 20px;
      border: 1px solid #dde7e2; background: #fff;
      transition: box-shadow .2s, transform .2s;
      height: 100%;
    }
    .agent-card:hover { box-shadow: 0 6px 20px rgba(0,0,0,.09); transform: translateY(-2px); }
    .agent-icon {
      width: 50px; height: 50px; border-radius: 12px;
      display: flex; align-items: center; justify-content: center;
      font-size: 1.4rem; margin-bottom: 12px;
    }
    .icon-green  { background: #e6f5ed; color: #1a7a4a; }
    .icon-blue   { background: #e6f0fb; color: #1a56ab; }
    .icon-red    { background: #fdecea; color: #c0392b; }
    .icon-orange { background: #fff3e0; color: #e67e00; }
    /* ── Forms ── */
    .form-label { font-size: .83rem; font-weight: 600; color: var(--text); margin-bottom: 4px; }
    .form-control, .form-select {
      border: 1px solid #cdd8d2; border-radius: 8px;
      font-size: .88rem; padding: 8px 12px;
    }
    .form-control:focus, .form-select:focus {
      border-color: var(--primary); box-shadow: 0 0 0 3px rgba(26,122,74,.12);
    }
    /* ── Buttons ── */
    .btn-primary-custom {
      background: var(--primary); color: #fff; border: none;
      border-radius: 8px; padding: 10px 22px; font-size: .88rem;
      font-weight: 600; cursor: pointer; transition: background .2s;
    }
    .btn-primary-custom:hover { background: var(--primary-d); }
    .btn-primary-custom:disabled { background: #aac7b8; cursor: not-allowed; }
    /* ── Response area ── */
    .response-box {
      background: #f8fdf9; border: 1px solid #c8e6d4;
      border-radius: 12px; padding: 20px 22px;
      font-size: .88rem; line-height: 1.75; color: var(--text);
      white-space: pre-wrap; min-height: 80px;
    }
    .response-box .thinking {
      color: var(--muted); display: flex; align-items: center; gap: 8px;
    }
    /* ── Spinner ── */
    .spinner-dots {
      display: inline-flex; gap: 5px; align-items: center;
    }
    .spinner-dots span {
      width: 7px; height: 7px; background: var(--primary);
      border-radius: 50%; animation: bounce 1.2s infinite ease-in-out;
    }
    .spinner-dots span:nth-child(2) { animation-delay: .2s; }
    .spinner-dots span:nth-child(3) { animation-delay: .4s; }
    @keyframes bounce {
      0%,80%,100% { transform: scale(0); }
      40%          { transform: scale(1); }
    }
    /* ── Hero section ── */
    .hero {
      background: linear-gradient(135deg, #0f2d1e 0%, #1a7a4a 100%);
      color: #fff; border-radius: 16px; padding: 40px 36px; margin-bottom: 28px;
    }
    .hero h1 { font-size: 1.7rem; font-weight: 700; margin-bottom: 8px; }
    .hero p  { color: #a8d5bc; font-size: .92rem; margin: 0; }
    .hero .stat-pill {
      background: rgba(255,255,255,.12); border-radius: 20px;
      padding: 6px 14px; font-size: .78rem; color: #d4f0e2;
      display: inline-block; margin: 4px 4px 0 0;
    }
    /* ── About page ── */
    .arch-step {
      display: flex; gap: 16px; align-items: flex-start;
      padding: 16px 0; border-bottom: 1px solid #edf2ee;
    }
    .arch-step:last-child { border-bottom: none; }
    .arch-num {
      width: 36px; height: 36px; border-radius: 50%;
      background: var(--primary); color: #fff;
      display: flex; align-items: center; justify-content: center;
      font-weight: 700; font-size: .9rem; flex-shrink: 0;
    }
    /* ── Condition checkboxes ── */
    .condition-grid {
      display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px;
    }
    .condition-item {
      border: 1px solid #dde7e2; border-radius: 10px; padding: 12px 14px;
      cursor: pointer; transition: border-color .2s, background .2s;
      display: flex; align-items: center; gap: 10px; font-size: .85rem;
    }
    .condition-item input[type=checkbox] { accent-color: var(--primary); width: 16px; height: 16px; }
    .condition-item:has(input:checked) { border-color: var(--primary); background: #edf7f1; }
    /* ── Chat interface ── */
    .chat-messages {
      background: #f8fdf9; border: 1px solid #c8e6d4;
      border-radius: 12px; padding: 20px; min-height: 200px;
      max-height: 520px; overflow-y: auto;
    }
    .msg { display: flex; gap: 12px; margin-bottom: 18px; }
    .msg.user  { flex-direction: row-reverse; }
    .msg-avatar {
      width: 36px; height: 36px; border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: .9rem; flex-shrink: 0;
    }
    .msg.bot  .msg-avatar { background: #e6f5ed; color: var(--primary); }
    .msg.user .msg-avatar { background: #e6f0fb; color: #1a56ab; }
    .msg-bubble {
      max-width: 80%; border-radius: 14px; padding: 12px 16px;
      font-size: .875rem; line-height: 1.65; white-space: pre-wrap;
    }
    .msg.bot  .msg-bubble { background: #fff; border: 1px solid #dde7e2; }
    .msg.user .msg-bubble { background: var(--primary); color: #fff; }
    /* ── Responsive sidebar toggle ── */
    #sidebarToggle {
      display: none; background: none; border: none;
      color: var(--text); font-size: 1.3rem; cursor: pointer;
    }
    @media (max-width: 768px) {
      #sidebar { transform: translateX(-100%); }
      #sidebar.open { transform: translateX(0); }
      #main { margin-left: 0; }
      #sidebarToggle { display: block; }
      .hero h1 { font-size: 1.3rem; }
    }
    /* ── Footer ── */
    footer {
      text-align: center; font-size: .72rem; color: var(--muted);
      padding: 16px; border-top: 1px solid #dde7e2;
      background: #fff;
    }
  </style>
</head>
<body>

<!-- ══════════════════════════ SIDEBAR ══════════════════════════ -->
<aside id="sidebar">
  <div class="brand">
    <h5><i class="bi bi-leaf-fill me-2" style="color:#4ecb82"></i>NutriWise AI</h5>
    <small>Powered by Google Gemini AI</small>
  </div>
  <nav>
    <div class="nav-label">Navigation</div>
    <a href="/"          class="{% if page=='home'    %}active{% endif %}"><i class="bi bi-house-door-fill"></i> Home</a>
    <a href="/chat"      class="{% if page=='chat'    %}active{% endif %}"><i class="bi bi-chat-dots-fill"></i> Nutrition Chat</a>
    <a href="/planner"   class="{% if page=='planner' %}active{% endif %}"><i class="bi bi-calendar2-check-fill"></i> Diet Planner</a>
    <a href="/advisor"   class="{% if page=='advisor' %}active{% endif %}"><i class="bi bi-heart-pulse-fill"></i> Health Advisor</a>
    <a href="/analyzer"  class="{% if page=='analyzer'%}active{% endif %}"><i class="bi bi-bar-chart-fill"></i> Meal Analyzer</a>
    <div class="nav-label" style="margin-top:12px">Info</div>
    <a href="/about"     class="{% if page=='about'   %}active{% endif %}"><i class="bi bi-info-circle-fill"></i> About</a>
  </nav>
  <div class="sidebar-footer">
    <span class="badge" style="background:#1a73e8;color:#fff;font-size:.68rem;padding:3px 8px;border-radius:4px">Google Gemini</span><br/>
    gemini-2.5-flash<br/>
    NutriWise AI v1.0
  </div>
</aside>

<!-- ══════════════════════════ MAIN ══════════════════════════ -->
<div id="main">
  <!-- Topbar -->
  <div id="topbar">
    <div style="display:flex;align-items:center;gap:12px">
      <button id="sidebarToggle" onclick="toggleSidebar()"><i class="bi bi-list"></i></button>
      <span class="page-title">{{ title }}</span>
    </div>
    <span class="badge-ibm" style="background:#1a73e8"><i class="bi bi-cpu-fill me-1"></i>Google Gemini AI</span>
  </div>

  <!-- Page content injected here -->
  <div class="content-area">
    {{ content | safe }}
  </div>

  <footer>Made with IBM Bob &nbsp;|&nbsp; NutriWise AI &copy; 2025 &nbsp;|&nbsp; Powered by Google Gemini AI</footer>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script>
function toggleSidebar() {
  document.getElementById('sidebar').classList.toggle('open');
}
function showLoading(boxId) {
  document.getElementById(boxId).innerHTML =
    '<div class="thinking"><div class="spinner-dots"><span></span><span></span><span></span></div>&nbsp;Google Gemini AI is thinking…</div>';
}
function renderMarkdown(text) {
  // Simple markdown-like rendering for bold and line breaks
  var bold   = new RegExp('\\\\*\\\\*([^*]+)\\\\*\\\\*', 'g');
  var italic = new RegExp('\\\\*([^*]+)\\\\*', 'g');
  return text
    .replace(bold,   '<strong>$1</strong>')
    .replace(italic, '<em>$1</em>')
    .replace(/^#{1,3} (.+)$/gm, '<strong style="font-size:1rem;color:#1a7a4a">$1</strong>')
    .replace(/^- (.+)$/gm,      '\u2022 $1')
    .replace(/\n/g, '<br/>');
}
</script>
{{ scripts | safe }}
</body>
</html>"""


# =============================================================================
# PAGE CONTENT TEMPLATES
# =============================================================================

HOME_CONTENT = """
<div class="hero">
  <h1><i class="bi bi-stars me-2"></i>Welcome to NutriWise AI</h1>
  <p>Your personalized AI nutrition coach powered by Google Gemini AI.<br>
     Four specialized AI agents work together to guide you towards better health.</p>
  <div style="margin-top:18px">
    <span class="stat-pill"><i class="bi bi-cpu me-1"></i>Google Gemini 2.5 Flash</span>
    <span class="stat-pill"><i class="bi bi-people me-1"></i>4 Specialized Agents</span>
    <span class="stat-pill"><i class="bi bi-shield-check me-1"></i>Evidence-Based AI</span>
    <span class="stat-pill"><i class="bi bi-lightning me-1"></i>Real-time Responses</span>
  </div>
</div>

<div class="row g-4">
  <!-- Agent 1 -->
  <div class="col-sm-6 col-xl-3">
    <a href="/chat" style="text-decoration:none">
      <div class="agent-card">
        <div class="agent-icon icon-green"><i class="bi bi-chat-dots-fill"></i></div>
        <h6 style="font-weight:700;margin-bottom:6px">Nutrition Chat</h6>
        <p style="font-size:.82rem;color:#6b7c72;margin:0">
          Ask any nutrition question and get instant AI-powered answers about foods, vitamins, and healthy eating.
        </p>
        <div style="margin-top:14px">
          <span class="badge" style="background:#e6f5ed;color:#1a7a4a;font-size:.7rem">Agent 1</span>
          <span class="badge" style="background:#f0f4ff;color:#1a56ab;font-size:.7rem;margin-left:4px">Knowledge</span>
        </div>
      </div>
    </a>
  </div>
  <!-- Agent 2 -->
  <div class="col-sm-6 col-xl-3">
    <a href="/planner" style="text-decoration:none">
      <div class="agent-card">
        <div class="agent-icon icon-blue"><i class="bi bi-calendar2-check-fill"></i></div>
        <h6 style="font-weight:700;margin-bottom:6px">Diet Planner</h6>
        <p style="font-size:.82rem;color:#6b7c72;margin:0">
          Get a personalized daily meal plan tailored to your age, weight, dietary preference, and fitness goals.
        </p>
        <div style="margin-top:14px">
          <span class="badge" style="background:#e6f0fb;color:#1a56ab;font-size:.7rem">Agent 2</span>
          <span class="badge" style="background:#f0f4ff;color:#1a56ab;font-size:.7rem;margin-left:4px">Planner</span>
        </div>
      </div>
    </a>
  </div>
  <!-- Agent 3 -->
  <div class="col-sm-6 col-xl-3">
    <a href="/advisor" style="text-decoration:none">
      <div class="agent-card">
        <div class="agent-icon icon-red"><i class="bi bi-heart-pulse-fill"></i></div>
        <h6 style="font-weight:700;margin-bottom:6px">Health Advisor</h6>
        <p style="font-size:.82rem;color:#6b7c72;margin:0">
          Receive condition-specific dietary guidance for Diabetes, PCOS, Hypertension, and more.
        </p>
        <div style="margin-top:14px">
          <span class="badge" style="background:#fdecea;color:#c0392b;font-size:.7rem">Agent 3</span>
          <span class="badge" style="background:#f0f4ff;color:#1a56ab;font-size:.7rem;margin-left:4px">Advisory</span>
        </div>
      </div>
    </a>
  </div>
  <!-- Agent 4 -->
  <div class="col-sm-6 col-xl-3">
    <a href="/analyzer" style="text-decoration:none">
      <div class="agent-card">
        <div class="agent-icon icon-orange"><i class="bi bi-bar-chart-fill"></i></div>
        <h6 style="font-weight:700;margin-bottom:6px">Meal Analyzer</h6>
        <p style="font-size:.82rem;color:#6b7c72;margin:0">
          Describe your meals in plain text and get an AI nutritional analysis with improvement suggestions.
        </p>
        <div style="margin-top:14px">
          <span class="badge" style="background:#fff3e0;color:#e67e00;font-size:.7rem">Agent 4</span>
          <span class="badge" style="background:#f0f4ff;color:#1a56ab;font-size:.7rem;margin-left:4px">Analyzer</span>
        </div>
      </div>
    </a>
  </div>
</div>

<div class="row g-4 mt-2">
  <div class="col-md-6">
    <div class="card p-0">
      <div class="card-header-custom">
        <i class="bi bi-cpu-fill"></i>
        <h5>Google Gemini Integration</h5>
      </div>
      <div class="card-body p-4">
        <p style="font-size:.88rem;color:#3d5248;margin-bottom:12px">
          NutriWise AI is built on top of Google Gemini AI. Every agent response
          is generated in real-time by gemini-2.5-flash, Google's fast and capable
          multimodal model available via Google AI Studio.
        </p>
        <ul style="font-size:.85rem;color:#3d5248;padding-left:20px;margin:0">
          <li>Real-time AI inference via Google Generative AI API</li>
          <li>Gemini 2.5 Flash for accurate, contextual nutrition advice</li>
          <li>Secure credential management via environment variables</li>
          <li>Multi-agent orchestration for specialised responses</li>
        </ul>
      </div>
    </div>
  </div>
  <div class="col-md-6">
    <div class="card p-0">
      <div class="card-header-custom" style="background:linear-gradient(135deg,#052e78,#1a56ab)">
        <i class="bi bi-lightning-fill"></i>
        <h5>Quick Start Guide</h5>
      </div>
      <div class="card-body p-4">
        <div style="font-size:.85rem;color:#3d5248">
          <div style="display:flex;gap:12px;margin-bottom:10px;align-items:center">
            <span style="background:#e6f5ed;color:#1a7a4a;width:26px;height:26px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.8rem;flex-shrink:0">1</span>
            Click <strong>Nutrition Chat</strong> to ask food &amp; nutrition questions
          </div>
          <div style="display:flex;gap:12px;margin-bottom:10px;align-items:center">
            <span style="background:#e6f0fb;color:#1a56ab;width:26px;height:26px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.8rem;flex-shrink:0">2</span>
            Use <strong>Diet Planner</strong> to generate your personalised meal plan
          </div>
          <div style="display:flex;gap:12px;margin-bottom:10px;align-items:center">
            <span style="background:#fdecea;color:#c0392b;width:26px;height:26px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.8rem;flex-shrink:0">3</span>
            Visit <strong>Health Advisor</strong> for condition-specific dietary guidance
          </div>
          <div style="display:flex;gap:12px;align-items:center">
            <span style="background:#fff3e0;color:#e67e00;width:26px;height:26px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.8rem;flex-shrink:0">4</span>
            Try <strong>Meal Analyzer</strong> to get feedback on your current meals
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
"""

CHAT_CONTENT = """
<div class="row g-4">
  <div class="col-lg-8">
    <div class="card p-0">
      <div class="card-header-custom">
        <i class="bi bi-chat-dots-fill"></i>
        <h5>Nutrition Knowledge Agent</h5>
        <span class="agent-badge">Agent 1</span>
      </div>
      <div class="card-body p-4">
        <p style="font-size:.85rem;color:#3d5248;margin-bottom:20px">
          Ask any nutrition question. Google Gemini AI will provide accurate, educational answers.
        </p>
        <!-- Chat messages window -->
        <div class="chat-messages" id="chatMessages">
          <div class="msg bot">
            <div class="msg-avatar"><i class="bi bi-leaf-fill"></i></div>
            <div class="msg-bubble">
              👋 Hello! I'm NutriWise AI, your personal nutrition coach powered by Google Gemini.
              Ask me anything about nutrition, food benefits, vitamins, or healthy eating!
            </div>
          </div>
        </div>
        <!-- Input -->
        <div style="display:flex;gap:10px;margin-top:16px">
          <input type="text" id="chatInput" class="form-control"
            placeholder="e.g. What are the benefits of oats?"
            onkeypress="if(event.key==='Enter') sendChat()"/>
          <button class="btn-primary-custom" onclick="sendChat()" id="chatBtn" style="white-space:nowrap">
            <i class="bi bi-send-fill me-1"></i>Ask
          </button>
        </div>
      </div>
    </div>
  </div>
  <div class="col-lg-4">
    <div class="card p-4" style="background:#f8fdf9">
      <h6 style="font-weight:700;margin-bottom:14px;color:#1a7a4a"><i class="bi bi-lightbulb-fill me-2"></i>Sample Questions</h6>
      <div style="display:flex;flex-direction:column;gap:8px">
        <button class="btn btn-outline-success btn-sm text-start" onclick="fillChat(this.textContent)" style="font-size:.8rem;border-radius:8px">What are the benefits of oats?</button>
        <button class="btn btn-outline-success btn-sm text-start" onclick="fillChat(this.textContent)" style="font-size:.8rem;border-radius:8px">Which foods are rich in protein?</button>
        <button class="btn btn-outline-success btn-sm text-start" onclick="fillChat(this.textContent)" style="font-size:.8rem;border-radius:8px">Is paneer healthy for muscle gain?</button>
        <button class="btn btn-outline-success btn-sm text-start" onclick="fillChat(this.textContent)" style="font-size:.8rem;border-radius:8px">What foods contain Vitamin B12?</button>
        <button class="btn btn-outline-success btn-sm text-start" onclick="fillChat(this.textContent)" style="font-size:.8rem;border-radius:8px">How much water should I drink daily?</button>
        <button class="btn btn-outline-success btn-sm text-start" onclick="fillChat(this.textContent)" style="font-size:.8rem;border-radius:8px">What are the best foods for iron?</button>
        <button class="btn btn-outline-success btn-sm text-start" onclick="fillChat(this.textContent)" style="font-size:.8rem;border-radius:8px">Is intermittent fasting healthy?</button>
      </div>
    </div>
  </div>
</div>
<script>
function fillChat(text) {
  document.getElementById('chatInput').value = text;
  sendChat();
}
function sendChat() {
  var input = document.getElementById('chatInput');
  var q = input.value.trim();
  if (!q) return;
  var msgs = document.getElementById('chatMessages');
  // Add user message
  msgs.innerHTML += '<div class="msg user"><div class="msg-avatar"><i class="bi bi-person-fill"></i></div><div class="msg-bubble">' + escapeHtml(q) + '</div></div>';
  input.value = '';
  // Add thinking bubble
  var thinkId = 'think_' + Date.now();
  msgs.innerHTML += '<div class="msg bot" id="' + thinkId + '"><div class="msg-avatar"><i class="bi bi-leaf-fill"></i></div><div class="msg-bubble"><div class="thinking"><div class="spinner-dots"><span></span><span></span><span></span></div>&nbsp;Google Gemini AI is thinking…</div></div></div>';
  msgs.scrollTop = msgs.scrollHeight;
  document.getElementById('chatBtn').disabled = true;
  fetch('/api/chat', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({question: q})
  })
  .then(r => r.json())
  .then(data => {
    var el = document.getElementById(thinkId);
    if (el) el.querySelector('.msg-bubble').innerHTML = renderMarkdown(data.response);
    msgs.scrollTop = msgs.scrollHeight;
    document.getElementById('chatBtn').disabled = false;
  })
  .catch(() => {
    var el = document.getElementById(thinkId);
    if (el) el.querySelector('.msg-bubble').textContent = 'Network error – make sure the Flask server is running on port 5000.';
    document.getElementById('chatBtn').disabled = false;
  });
}
function escapeHtml(t) {
  return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
</script>
"""

PLANNER_CONTENT = """
<div class="row g-4">
  <div class="col-lg-5">
    <div class="card p-0">
      <div class="card-header-custom" style="background:linear-gradient(135deg,#1a56ab,#0d3d7c)">
        <i class="bi bi-person-vcard-fill"></i>
        <h5>Your Profile</h5>
        <span class="agent-badge">Agent 2</span>
      </div>
      <div class="card-body p-4">
        <div class="row g-3">
          <div class="col-6">
            <label class="form-label">Age (years)</label>
            <input type="number" id="age" class="form-control" value="28" min="10" max="100"/>
          </div>
          <div class="col-6">
            <label class="form-label">Gender</label>
            <select id="gender" class="form-select">
              <option>Male</option>
              <option>Female</option>
              <option>Other</option>
            </select>
          </div>
          <div class="col-6">
            <label class="form-label">Height (cm)</label>
            <input type="number" id="height" class="form-control" value="170" min="100" max="250"/>
          </div>
          <div class="col-6">
            <label class="form-label">Weight (kg)</label>
            <input type="number" id="weight" class="form-control" value="70" min="20" max="300"/>
          </div>
          <div class="col-12">
            <label class="form-label">Dietary Preference</label>
            <select id="dietary_pref" class="form-select">
              <option>Omnivore</option>
              <option>Vegetarian</option>
              <option>Vegan</option>
              <option>Pescatarian</option>
              <option>Keto</option>
              <option>Mediterranean</option>
            </select>
          </div>
          <div class="col-12">
            <label class="form-label">Activity Level</label>
            <select id="activity_level" class="form-select">
              <option>Sedentary</option>
              <option>Lightly Active</option>
              <option selected>Moderately Active</option>
              <option>Very Active</option>
              <option>Extremely Active</option>
            </select>
          </div>
          <div class="col-12">
            <label class="form-label">Fitness Goal</label>
            <select id="goal" class="form-select">
              <option>Weight Loss</option>
              <option>Weight Gain</option>
              <option>Muscle Gain</option>
              <option selected>General Wellness</option>
            </select>
          </div>
          <div class="col-12 pt-2">
            <button class="btn-primary-custom w-100" onclick="generatePlan()" id="planBtn">
              <i class="bi bi-calendar2-check-fill me-2"></i>Generate My Meal Plan
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
  <div class="col-lg-7">
    <div class="card p-0 h-100">
      <div class="card-header-custom" style="background:linear-gradient(135deg,#1a56ab,#0d3d7c)">
        <i class="bi bi-clipboard2-data-fill"></i>
        <h5>Your Personalised Meal Plan</h5>
      </div>
      <div class="card-body p-4">
        <div class="response-box" id="planResult">
          <span style="color:#8aa898;font-size:.85rem">
            <i class="bi bi-info-circle me-2"></i>Fill in your profile and click "Generate My Meal Plan" to get an AI-powered diet plan from Google Gemini.
          </span>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
function generatePlan() {
  var btn = document.getElementById('planBtn');
  btn.disabled = true;
  btn.innerHTML = '<div class="spinner-dots" style="display:inline-flex"><span></span><span></span><span></span></div> Generating…';
  document.getElementById('planResult').innerHTML = '<div class="thinking"><div class="spinner-dots"><span></span><span></span><span></span></div>&nbsp;Google Gemini AI is creating your personalised meal plan…</div>';
  fetch('/api/planner', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({
      age:           parseInt(document.getElementById('age').value),
      gender:        document.getElementById('gender').value,
      height:        parseFloat(document.getElementById('height').value),
      weight:        parseFloat(document.getElementById('weight').value),
      dietary_pref:  document.getElementById('dietary_pref').value,
      activity_level:document.getElementById('activity_level').value,
      goal:          document.getElementById('goal').value
    })
  })
  .then(r => r.json())
  .then(data => {
    document.getElementById('planResult').innerHTML = renderMarkdown(data.response);
    btn.disabled = false;
    btn.innerHTML = '<i class="bi bi-calendar2-check-fill me-2"></i>Generate My Meal Plan';
  })
  .catch(() => {
    document.getElementById('planResult').textContent = 'Network error – make sure the Flask server is running on port 5000.';
    btn.disabled = false;
    btn.innerHTML = '<i class="bi bi-calendar2-check-fill me-2"></i>Generate My Meal Plan';
  });
}
</script>
"""

ADVISOR_CONTENT = """
<div class="row g-4">
  <div class="col-lg-5">
    <div class="card p-0">
      <div class="card-header-custom" style="background:linear-gradient(135deg,#c0392b,#962d22)">
        <i class="bi bi-heart-pulse-fill"></i>
        <h5>Health Condition Selector</h5>
        <span class="agent-badge">Agent 3</span>
      </div>
      <div class="card-body p-4">
        <p style="font-size:.84rem;color:#3d5248;margin-bottom:18px">
          Select one or more health conditions to receive tailored dietary advice from Google Gemini AI.
        </p>
        <div class="condition-grid">
          <label class="condition-item"><input type="checkbox" name="cond" value="Diabetes"/> <span>🩸 Diabetes</span></label>
          <label class="condition-item"><input type="checkbox" name="cond" value="Hypertension"/> <span>💓 Hypertension</span></label>
          <label class="condition-item"><input type="checkbox" name="cond" value="Obesity"/> <span>⚖️ Obesity</span></label>
          <label class="condition-item"><input type="checkbox" name="cond" value="Heart Disease"/> <span>❤️ Heart Disease</span></label>
          <label class="condition-item"><input type="checkbox" name="cond" value="PCOS"/> <span>🌸 PCOS</span></label>
          <label class="condition-item"><input type="checkbox" name="cond" value="High Cholesterol"/> <span>🧪 High Cholesterol</span></label>
          <label class="condition-item"><input type="checkbox" name="cond" value="Thyroid Disorders"/> <span>🦋 Thyroid</span></label>
          <label class="condition-item"><input type="checkbox" name="cond" value="Anaemia"/> <span>💊 Anaemia</span></label>
        </div>
        <button class="btn-primary-custom w-100 mt-4" onclick="getAdvice()" id="adviceBtn" style="background:#c0392b">
          <i class="bi bi-heart-pulse-fill me-2"></i>Get Health Advice
        </button>
      </div>
    </div>
  </div>
  <div class="col-lg-7">
    <div class="card p-0 h-100">
      <div class="card-header-custom" style="background:linear-gradient(135deg,#c0392b,#962d22)">
        <i class="bi bi-clipboard2-heart-fill"></i>
        <h5>Personalised Health Recommendations</h5>
      </div>
      <div class="card-body p-4">
        <div class="response-box" id="adviceResult">
          <span style="color:#8aa898;font-size:.85rem">
            <i class="bi bi-info-circle me-2"></i>Select health conditions and click "Get Health Advice" to receive AI-powered dietary recommendations from Google Gemini.
          </span>
        </div>
        <div style="margin-top:14px;padding:12px 14px;background:#fff8f7;border:1px solid #f5c6c2;border-radius:8px;font-size:.78rem;color:#8b3a35">
          <i class="bi bi-exclamation-triangle-fill me-1"></i>
          <strong>Disclaimer:</strong> Educational information only. Consult a healthcare professional for medical advice.
        </div>
      </div>
    </div>
  </div>
</div>
<script>
function getAdvice() {
  var checked = Array.from(document.querySelectorAll('input[name=cond]:checked')).map(e => e.value);
  if (checked.length === 0) {
    alert('Please select at least one health condition.');
    return;
  }
  var btn = document.getElementById('adviceBtn');
  btn.disabled = true;
  btn.innerHTML = '<div class="spinner-dots" style="display:inline-flex"><span></span><span></span><span></span></div> Generating…';
  document.getElementById('adviceResult').innerHTML = '<div class="thinking"><div class="spinner-dots"><span></span><span></span><span></span></div>&nbsp;Google Gemini AI is generating health recommendations…</div>';
  fetch('/api/advisor', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({conditions: checked})
  })
  .then(r => r.json())
  .then(data => {
    document.getElementById('adviceResult').innerHTML = renderMarkdown(data.response);
    btn.disabled = false;
    btn.innerHTML = '<i class="bi bi-heart-pulse-fill me-2"></i>Get Health Advice';
  })
  .catch(() => {
    document.getElementById('adviceResult').textContent = 'Error connecting to Google Gemini. Please check your GOOGLE_API_KEY.';
    btn.disabled = false;
    btn.innerHTML = '<i class="bi bi-heart-pulse-fill me-2"></i>Get Health Advice';
  });
}
</script>
"""

ANALYZER_CONTENT = """
<div class="row g-4">
  <div class="col-lg-5">
    <div class="card p-0">
      <div class="card-header-custom" style="background:linear-gradient(135deg,#e67e00,#b36200)">
        <i class="bi bi-pencil-square"></i>
        <h5>Enter Your Meals</h5>
        <span class="agent-badge">Agent 4</span>
      </div>
      <div class="card-body p-4">
        <p style="font-size:.84rem;color:#3d5248;margin-bottom:16px">
          Describe your meals in plain text. Google Gemini AI will analyse the nutritional quality and suggest improvements.
        </p>
        <label class="form-label">Your Daily Meals</label>
        <textarea id="mealInput" class="form-control" rows="12"
          placeholder="Example:&#10;&#10;Breakfast:&#10;  2 Rotis&#10;  Dal with tadka&#10;  1 cup chai with sugar&#10;&#10;Lunch:&#10;  Rice (2 cups)&#10;  Paneer Curry&#10;  Raita&#10;&#10;Snack:&#10;  Samosa&#10;  Tea&#10;&#10;Dinner:&#10;  Mixed Salad&#10;  Grilled Chicken (150g)&#10;  Brown Rice"></textarea>
        <button class="btn-primary-custom w-100 mt-3" onclick="analyzeMeal()" id="analyzeBtn" style="background:#e67e00">
          <i class="bi bi-bar-chart-fill me-2"></i>Analyse My Meals
        </button>
      </div>
    </div>
  </div>
  <div class="col-lg-7">
    <div class="card p-0 h-100">
      <div class="card-header-custom" style="background:linear-gradient(135deg,#e67e00,#b36200)">
        <i class="bi bi-graph-up-arrow"></i>
        <h5>Meal Analysis Report</h5>
      </div>
      <div class="card-body p-4">
        <div class="response-box" id="analyzeResult">
          <span style="color:#8aa898;font-size:.85rem">
            <i class="bi bi-info-circle me-2"></i>Enter your meals and click "Analyse My Meals" to get an AI nutritional analysis from Google Gemini.
          </span>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
function analyzeMeal() {
  var meal = document.getElementById('mealInput').value.trim();
  if (!meal) {
    alert('Please enter your meal description.');
    return;
  }
  var btn = document.getElementById('analyzeBtn');
  btn.disabled = true;
  btn.innerHTML = '<div class="spinner-dots" style="display:inline-flex"><span></span><span></span><span></span></div> Analysing…';
  document.getElementById('analyzeResult').innerHTML = '<div class="thinking"><div class="spinner-dots"><span></span><span></span><span></span></div>&nbsp;Google Gemini AI is analysing your meal plan…</div>';
  fetch('/api/analyzer', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({meal_description: meal})
  })
  .then(r => r.json())
  .then(data => {
    document.getElementById('analyzeResult').innerHTML = renderMarkdown(data.response);
    btn.disabled = false;
    btn.innerHTML = '<i class="bi bi-bar-chart-fill me-2"></i>Analyse My Meals';
  })
  .catch(() => {
    document.getElementById('analyzeResult').textContent = 'Network error – make sure the Flask server is running on port 5000.';
    btn.disabled = false;
    btn.innerHTML = '<i class="bi bi-bar-chart-fill me-2"></i>Analyse My Meals';
  });
}
</script>
"""

ABOUT_CONTENT = """
<div class="row g-4">
  <div class="col-12">
    <div class="hero" style="margin-bottom:0">
      <h1><i class="bi bi-info-circle-fill me-2"></i>About NutriWise AI</h1>
      <p>A multi-agent AI nutrition assistant built on Google Gemini AI,
         showcasing Agentic AI architecture for healthcare and wellness applications.</p>
    </div>
  </div>

  <div class="col-lg-6">
    <div class="card p-0">
      <div class="card-header-custom">
        <i class="bi bi-diagram-3-fill"></i>
        <h5>Multi-Agent Architecture</h5>
      </div>
      <div class="card-body p-4">
        <div class="arch-step">
          <div class="arch-num">1</div>
          <div>
            <strong>Nutrition Knowledge Agent</strong>
            <p style="font-size:.83rem;color:#6b7c72;margin:4px 0 0">
              Answers general nutrition questions. Provides educational information about foods, vitamins, minerals, and dietary concepts using Google Gemini's knowledge base.
            </p>
          </div>
        </div>
        <div class="arch-step">
          <div class="arch-num" style="background:#1a56ab">2</div>
          <div>
            <strong>Diet Planner Agent</strong>
            <p style="font-size:.83rem;color:#6b7c72;margin:4px 0 0">
              Generates personalised daily meal plans. Takes user profile (age, weight, goals, dietary preferences) and creates structured breakfast, lunch, snack, and dinner recommendations with calorie targets.
            </p>
          </div>
        </div>
        <div class="arch-step">
          <div class="arch-num" style="background:#c0392b">3</div>
          <div>
            <strong>Health Advisory Agent</strong>
            <p style="font-size:.83rem;color:#6b7c72;margin:4px 0 0">
              Provides condition-specific dietary guidance for Diabetes, Hypertension, PCOS, Heart Disease, and more. Always includes a medical disclaimer to promote safe usage.
            </p>
          </div>
        </div>
        <div class="arch-step">
          <div class="arch-num" style="background:#e67e00">4</div>
          <div>
            <strong>Meal Analysis Agent</strong>
            <p style="font-size:.83rem;color:#6b7c72;margin:4px 0 0">
              Analyses free-text meal descriptions to estimate nutritional quality, identify strengths and deficiencies, and suggest practical improvements and healthier alternatives.
            </p>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="col-lg-6">
    <div class="card p-0 mb-4">
      <div class="card-header-custom" style="background:linear-gradient(135deg,#052e78,#1a56ab)">
        <i class="bi bi-cpu-fill"></i>
        <h5>Google Gemini Integration</h5>
      </div>
      <div class="card-body p-4">
        <div style="font-size:.85rem;color:#3d5248;line-height:1.8">
          <p><strong>Model:</strong> gemini-2.5-flash</p>
          <p><strong>Platform:</strong> Google AI Studio / Gemini API</p>
          <p><strong>Integration:</strong> google-generativeai Python SDK</p>
          <p><strong>Pattern:</strong> Single <code>generate_response()</code> function shared by all agents</p>
          <p><strong>Auth:</strong> GOOGLE_API_KEY via environment variable</p>
          <p style="margin:0"><strong>Parameters:</strong> temperature=0.7, top_p=0.9, max_output_tokens=800–1200</p>
        </div>
      </div>
    </div>
    <div class="card p-0">
      <div class="card-header-custom" style="background:linear-gradient(135deg,#2d6a4f,#1a7a4a)">
        <i class="bi bi-gear-fill"></i>
        <h5>Orchestrator Flow</h5>
      </div>
      <div class="card-body p-4">
        <div style="font-size:.84rem;color:#3d5248;line-height:1.9">
          <div>① User selects a feature (Chat / Planner / Advisor / Analyzer)</div>
          <div style="padding-left:16px;color:#6b7c72">↓ HTTP POST to Flask API endpoint</div>
          <div>② Orchestrator receives request + payload</div>
          <div style="padding-left:16px;color:#6b7c72">↓ Routes to the correct agent function</div>
          <div>③ Agent builds a domain-specific prompt</div>
          <div style="padding-left:16px;color:#6b7c72">↓ Calls <code>generate_response(prompt)</code></div>
          <div>④ Google Gemini model is invoked</div>
          <div style="padding-left:16px;color:#6b7c72">↓ Response returned as JSON</div>
          <div>⑤ UI renders the Gemini response to the user</div>
        </div>
      </div>
    </div>
  </div>

  <div class="col-12">
    <div class="card p-4" style="background:#f8fdf9;border-color:#c8e6d4">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px">
        <i class="bi bi-trophy-fill" style="color:#e67e00;font-size:1.3rem"></i>
        <h6 style="margin:0;font-weight:700">Use Cases &amp; Applications</h6>
      </div>
      <div class="row g-3">
        <div class="col-md-3">
          <div style="background:#fff;border:1px solid #dde7e2;border-radius:10px;padding:14px;font-size:.83rem;text-align:center">
            <i class="bi bi-building" style="color:#052e78;font-size:1.3rem;display:block;margin-bottom:8px"></i>
            <strong>AI Hackathons</strong><br/>
            <span style="color:#6b7c72">Agentic AI demonstration with Google Gemini integration</span>
          </div>
        </div>
        <div class="col-md-3">
          <div style="background:#fff;border:1px solid #dde7e2;border-radius:10px;padding:14px;font-size:.83rem;text-align:center">
            <i class="bi bi-mortarboard-fill" style="color:#1a7a4a;font-size:1.3rem;display:block;margin-bottom:8px"></i>
            <strong>SkillsBuild</strong><br/>
            <span style="color:#6b7c72">SkillsBuild showcase for AI in healthcare</span>
          </div>
        </div>
        <div class="col-md-3">
          <div style="background:#fff;border:1px solid #dde7e2;border-radius:10px;padding:14px;font-size:.83rem;text-align:center">
            <i class="bi bi-laptop-fill" style="color:#c0392b;font-size:1.3rem;display:block;margin-bottom:8px"></i>
            <strong>College Projects</strong><br/>
            <span style="color:#6b7c72">AI/ML final year project with real LLM integration</span>
          </div>
        </div>
        <div class="col-md-3">
          <div style="background:#fff;border:1px solid #dde7e2;border-radius:10px;padding:14px;font-size:.83rem;text-align:center">
            <i class="bi bi-presentation-fill" style="color:#e67e00;font-size:1.3rem;display:block;margin-bottom:8px"></i>
            <strong>AI Demos</strong><br/>
            <span style="color:#6b7c72">Live multi-agent AI demonstration for presentations</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
"""


# =============================================================================
# FLASK ROUTES
# =============================================================================

def render_page(page: str, title: str, content: str, scripts: str = "") -> str:
    """Helper: renders the full page using the base template via render_template_string."""
    return render_template_string(
        BASE_HTML,
        page=page,
        title=title,
        content=content,
        scripts=scripts
    )


@app.route("/")
def home():
    """Home page – overview of NutriWise AI."""
    return render_page("home", "Home – NutriWise AI", HOME_CONTENT)


@app.route("/chat")
def chat():
    """Nutrition Chat page – Agent 1: Nutrition Knowledge Agent."""
    return render_page("chat", "Nutrition Chat – Agent 1", CHAT_CONTENT)


@app.route("/planner")
def planner():
    """Diet Planner page – Agent 2: Diet Planner Agent."""
    return render_page("planner", "Diet Planner – Agent 2", PLANNER_CONTENT)


@app.route("/advisor")
def advisor():
    """Health Advisor page – Agent 3: Health Advisory Agent."""
    return render_page("advisor", "Health Advisor – Agent 3", ADVISOR_CONTENT)


@app.route("/analyzer")
def analyzer():
    """Meal Analyzer page – Agent 4: Meal Analysis Agent."""
    return render_page("analyzer", "Meal Analyzer – Agent 4", ANALYZER_CONTENT)


@app.route("/about")
def about():
    """About page – Architecture and Google Gemini integration details."""
    return render_page("about", "About NutriWise AI", ABOUT_CONTENT)


# =============================================================================
# API ENDPOINTS – These call the Orchestrator → Agent → Google Gemini
# =============================================================================

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    API endpoint for Agent 1 (Nutrition Knowledge).
    Receives: { "question": "..." }
    Returns:  { "response": "..." }
    """
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Question is required"}), 400

    # Route through orchestrator → nutrition_knowledge_agent → Google Gemini
    response = orchestrator("nutrition_knowledge", {"question": question})
    return jsonify({"response": response})


@app.route("/api/planner", methods=["POST"])
def api_planner():
    """
    API endpoint for Agent 2 (Diet Planner).
    Receives: { age, gender, height, weight, dietary_pref, activity_level, goal }
    Returns:  { "response": "..." }
    """
    data = request.get_json(force=True)

    # Route through orchestrator → diet_planner_agent → Google Gemini
    response = orchestrator("diet_planner", {
        "age":            data.get("age", 25),
        "gender":         data.get("gender", "Not specified"),
        "height":         data.get("height", 170),
        "weight":         data.get("weight", 70),
        "dietary_pref":   data.get("dietary_pref", "Omnivore"),
        "activity_level": data.get("activity_level", "Moderate"),
        "goal":           data.get("goal", "General Wellness")
    })
    return jsonify({"response": response})


@app.route("/api/advisor", methods=["POST"])
def api_advisor():
    """
    API endpoint for Agent 3 (Health Advisory).
    Receives: { "conditions": ["Diabetes", ...] }
    Returns:  { "response": "..." }
    """
    data = request.get_json(force=True)
    conditions = data.get("conditions", ["General Wellness"])

    # Route through orchestrator → health_advisory_agent → Google Gemini
    response = orchestrator("health_advisory", {"conditions": conditions})
    return jsonify({"response": response})


@app.route("/api/analyzer", methods=["POST"])
def api_analyzer():
    """
    API endpoint for Agent 4 (Meal Analysis).
    Receives: { "meal_description": "..." }
    Returns:  { "response": "..." }
    """
    data = request.get_json(force=True)
    meal_description = data.get("meal_description", "").strip()
    if not meal_description:
        return jsonify({"error": "Meal description is required"}), 400

    # Route through orchestrator → meal_analysis_agent → Google Gemini
    response = orchestrator("meal_analysis", {"meal_description": meal_description})
    return jsonify({"response": response})


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  NutriWise AI – Personalized Nutrition Coach")
    print("  Powered by Google Gemini AI")
    print("=" * 60)
    print(f"  Model   : {GEMINI_MODEL_ID}")
    print(f"  API Key : {'[SET]' if GOOGLE_API_KEY != 'your-google-api-key-here' else '[NOT SET]'}")
    print("=" * 60)
    print("  Open http://127.0.0.1:5000 in your browser")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
