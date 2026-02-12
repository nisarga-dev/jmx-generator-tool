import streamlit as st
import requests
import base64
import json
import time
from io import BytesIO
from PIL import Image
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
import streamlit.components.v1 as components

# ===============================
# CONFIG
# ===============================
IMGBB_API_KEY = "a34d1a85a247c47cd916c8a2d5848c05"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ===============================
# HELPERS (UNCHANGED)
# ===============================

def compress_image_bytes(uploaded_file, target_kb=400):
    img = Image.open(uploaded_file).convert("RGB")
    buffer = BytesIO()
    quality = 85
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    while buffer.tell() > target_kb * 1024 and quality > 30:
        quality -= 5
        buffer.seek(0)
        buffer.truncate(0)
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def upload_to_imgbb(image_bytes):
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    res = requests.post(
        "https://api.imgbb.com/1/upload",
        data={"key": IMGBB_API_KEY, "image": encoded},
        timeout=30
    )
    if res.status_code == 200:
        return res.json()["data"]["url"]
    st.error("ImgBB upload failed")
    return None


def call_groq_vision(groq_key, image_url):
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json"
    }

    prompt_text = (
        "Extract API interactions from the diagram and return ONLY a JSON object:"
        '{ "actors": [...], "apis": [ { "name":"GetUser", "method":"GET", "url":"https://x.com/u" } ] }'
    )

    payload = {
        "model": GROQ_VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }

    res = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
    return res.json()["choices"][0]["message"]["content"]


def ensure_apis_struct(vision_json):
    apis_raw = vision_json.get("apis") or []
    actors = vision_json.get("actors") or []

    fixed = []
    for api in apis_raw:
        fixed.append({
            "name": api.get("name", "API"),
            "method": api.get("method", "GET").upper(),
            "url": api.get("url", ""),
            "body_schema": api.get("body_schema")
        })

    return {"actors": actors, "apis": fixed}


# ===============================
# FUNCTIONAL TESTCASES (IMPROVED, NOT SIMPLIFIED)
# ===============================

def generate_functional_testcases(apis):
    tcs = []
    tid = 1

    for api in apis:
        name = api["name"]
        method = api["method"]
        url = api["url"]

        tcs.append({
            "ID": f"TC{tid:02d}",
            "Scenario": f"Valid {method} request to {name} API",
            "Input": api,
            "Expected Output": {"status": 200},
            "Test Type": "Positive",
            "Remarks": (
                "1. Client sends a valid request with all required parameters.\n"
                "2. Server validates request format and parameters.\n"
                "3. Business logic is executed successfully.\n"
                "4. Server returns HTTP 200 with valid response body."
            )
        })
        tid += 1

        tcs.append({
            "ID": f"TC{tid:02d}",
            "Scenario": f"Invalid request to {name} API",
            "Input": api,
            "Expected Output": {"status": 400},
            "Test Type": "Negative",
            "Remarks": (
                "1. Client sends request with missing or invalid parameters.\n"
                "2. Server detects validation error.\n"
                "3. Request is rejected.\n"
                "4. Server returns HTTP 400 error response."
            )
        })
        tid += 1

    return tcs


def functional_testcases_to_md(tcs):
    md = "# Functional Test Cases\n\n"
    md += "| ID | Scenario | Type | Remarks |\n"
    md += "|---|---|---|---|\n"
    for tc in tcs:
        md += f"| {tc['ID']} | {tc['Scenario']} | {tc['Test Type']} | {tc['Remarks']} |\n"
    return md


# ===============================
# VISUALIZER (POSITIVE FLOW ONLY)
# ===============================

def render_positive_flow_diagram(testcases):
    positives = [tc for tc in testcases if tc.get("Test Type") == "Positive"]
    if not positives:
        return

    nodes = [{"data": {"id": "start", "label": "Start"}}]
    edges = []
    prev = "start"

    for tc in positives:
        nodes.append({
            "data": {
                "id": tc["ID"],
                "label": f"{tc['ID']}\n{tc['Scenario']}"
            }
        })
        edges.append({
            "data": {
                "source": prev,
                "target": tc["ID"]
            }
        })
        prev = tc["ID"]

    nodes.append({
        "data": {
            "id": "success",
            "label": "Success"
        }
    })
    edges.append({
        "data": {
            "source": prev,
            "target": "success"
        }
    })

    elements = nodes + edges

    html = f"""
    <html>
    <head>
      <script src="https://unpkg.com/cytoscape/dist/cytoscape.min.js"></script>
      <style>
        #cy {{
          width: 100%;
          height: 550px;
          border: 1px solid #ddd;
        }}
      </style>
    </head>
    <body>
      <div id="cy"></div>
      <script>
        var cy = cytoscape({{
          container: document.getElementById('cy'),
          elements: {json.dumps(elements)},
          layout: {{
            name: 'cose',
            animate: true
          }},
          userZoomingEnabled: true,
          userPanningEnabled: true,
          style: [
            {{
              selector: 'node',
              style: {{
                'label': 'data(label)',
                'background-color': '#2196F3',
                'color': '#fff',
                'text-valign': 'center',
                'text-halign': 'center',
                'font-size': '12px'
              }}
            }},
            {{
              selector: 'node:hover',
              style: {{
                'background-color': '#FF9800',
                'width': 60,
                'height': 60
              }}
            }},
            {{
              selector: 'edge',
              style: {{
                'width': 3,
                'line-color': '#9E9E9E',
                'target-arrow-shape': 'triangle',
                'target-arrow-color': '#9E9E9E'
              }}
            }}
          ]
        }});
      </script>
    </body>
    </html>
    """

    st.subheader("Positive Functional Flow Diagram")
    components.html(html, height=600, scrolling=False)


# ===============================
# JMX BUILDER (ORIGINAL STRUCTURE + SAFE LISTENER)
# ===============================

def url_to_parts(url):
    p = urlparse(url)
    return p.scheme or "https", p.hostname or "", str(p.port or 443), p.path or "/"


def build_jmx_from_yaml(y):
    """100% minimal valid JMX (original structure + View Results Tree)."""

    jmeter = ET.Element(
        "jmeterTestPlan",
        attrib={"version": "1.2", "properties": "5.0", "jmeter": "5.6.3"}
    )
    root_ht = ET.SubElement(jmeter, "hashTree")

    # Test Plan
    tp = ET.SubElement(
        root_ht, "TestPlan",
        guiclass="TestPlanGui",
        testclass="TestPlan",
        testname="Simple Test Plan",
        enabled="true"
    )

    ET.SubElement(
        tp,
        "elementProp",
        name="TestPlan.user_defined_variables",
        elementType="Arguments"
    )

    tp_ht = ET.SubElement(root_ht, "hashTree")

    # Thread Group
    tg = ET.SubElement(
        tp_ht, "ThreadGroup",
        guiclass="ThreadGroupGui",
        testclass="ThreadGroup",
        testname="Thread Group",
        enabled="true"
    )

    ET.SubElement(tg, "intProp", name="ThreadGroup.num_threads").text = "1"
    ET.SubElement(tg, "intProp", name="ThreadGroup.ramp_time").text = "1"
    ET.SubElement(tg, "boolProp", name="ThreadGroup.same_user_on_next_iteration").text = "true"
    ET.SubElement(tg, "stringProp", name="ThreadGroup.on_sample_error").text = "continue"

    loop_ctrl = ET.SubElement(
        tg,
        "elementProp",
        name="ThreadGroup.main_controller",
        elementType="LoopController",
        guiclass="LoopControlPanel",
        testclass="LoopController",
        testname="Loop Controller"
    )

    ET.SubElement(loop_ctrl, "stringProp", name="LoopController.loops").text = "1"
    ET.SubElement(loop_ctrl, "boolProp", name="LoopController.continue_forever").text = "false"

    tg_ht = ET.SubElement(tp_ht, "hashTree")

    # HTTP Samplers (UNCHANGED)
    for req in y["requests"]:
        proto, dom, port, path = url_to_parts(req["url"])

        sampler = ET.SubElement(
            tg_ht, "HTTPSamplerProxy",
            guiclass="HttpTestSampleGui",
            testclass="HTTPSamplerProxy",
            testname=req["name"],
            enabled="true"
        )

        ET.SubElement(sampler, "stringProp", name="HTTPSampler.domain").text = dom
        ET.SubElement(sampler, "stringProp", name="HTTPSampler.port").text = port
        ET.SubElement(sampler, "stringProp", name="HTTPSampler.protocol").text = proto
        ET.SubElement(sampler, "stringProp", name="HTTPSampler.path").text = path
        ET.SubElement(sampler, "stringProp", name="HTTPSampler.method").text = req["method"]

        # REQUIRED hashTree after sampler
        ET.SubElement(tg_ht, "hashTree")

    # ✅ View Results Tree (SAFE ADDITION)
    listener = ET.SubElement(
        tg_ht,
        "ResultCollector",
        guiclass="ViewResultsFullVisualizer",
        testclass="ResultCollector",
        testname="View Results Tree",
        enabled="true"
    )

    ET.SubElement(listener, "boolProp", name="ResultCollector.error_logging").text = "false"
    ET.SubElement(listener, "stringProp", name="filename").text = ""

    # REQUIRED hashTree after listener
    ET.SubElement(tg_ht, "hashTree")

    # Pretty print
    rough = ET.tostring(jmeter, "utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ")
    return pretty


# ===============================
# STREAMLIT UI (UNCHANGED FLOW)
# ===============================

st.set_page_config(page_title="Functional + Load Testing Generator", layout="wide")
st.title("Functional + Load Testing Generator (Groq Vision + JMX)")

groq_key = st.text_input("Groq API Key", type="password")
uploaded = st.file_uploader("Upload Sequence Diagram", type=["png","jpg","jpeg"])

if uploaded:
    st.image(uploaded, use_column_width=True)

if uploaded and groq_key and st.button("Generate Testcases"):
    bytes_img = compress_image_bytes(uploaded)
    url = upload_to_imgbb(bytes_img)

    vision_res = call_groq_vision(groq_key, url)
    vision_res = json.loads(vision_res)

    norm = ensure_apis_struct(vision_res)
    apis = norm["apis"]

    st.subheader("Extracted APIs")
    st.json(norm)

    tcs = generate_functional_testcases(apis)
    md = functional_testcases_to_md(tcs)

    st.subheader("Functional Testcases")
    st.markdown(md)

    render_positive_flow_diagram(tcs)

    y = {"requests": apis}
    jmx = build_jmx_from_yaml(y)

    st.download_button("Download Functional Testcases (MD)", md, file_name="testcases.md")
    st.download_button("Download JMX Test Plan", jmx, file_name="plan.jmx")

    st.success("Done.")
