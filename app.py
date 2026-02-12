import streamlit as st
import requests
import base64
import json
from io import BytesIO
from PIL import Image
import yaml
from rag.rag_engine import query_docs


# ===============================
# SSL FIX (GLOBAL – SAFE FOR DEV)
# ===============================
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
SESSION = requests.Session()
SESSION.verify = False

# ===============================
# CONFIG
# ===============================
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ===============================
# HELPERS
# ===============================
def compress_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()

def image_to_base64(img_bytes):
    return base64.b64encode(img_bytes).decode("utf-8")

# ✅ DEDUPLICATION LOGIC (ADDED)
def deduplicate_apis(apis):
    seen = set()
    final = []

    for api in apis:
        method = api.get("method")
        path = api.get("path", "").strip().lower()

        # ❌ Ignore acknowledgements ONLY
        if "ack" in path:
            continue

        key = (method, path)
        if key not in seen:
            seen.add(key)
            final.append(api)

    return final


def extract_apis_from_diagram(api_key, image_bytes):
    base64_img = image_to_base64(image_bytes)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    prompt = """
You are given a SEQUENCE DIAGRAM.

TASK: Extract ONLY the BUSINESS API calls present in the diagram.

Definition of BUSINESS API:
- Initiates a business operation
- Requests or modifies data
- Has a meaningful request/response payload

For EACH BUSINESS API return:
- method (GET, POST, PUT, DELETE)
- path (exact URL path)
- request_fields (array of strings)
- response_status (array of integers)

DO NOT extract the following:
- Acknowledgements or ACK messages
- Success/Failure notifications without payload
- Protocol-level confirmations
- Internal handshakes

IMPORTANT MESSAGE-LEVEL RULE: Treat REQUEST and RESPONSE messages as SEPARATE APIs.

Examples:
- ReqListAccount → API
- RespListAccount → API

DO NOT merge request and response into a single API.

IGNORE ONLY:
- Acknowledgements / ACK messages
- Internal forwarding or renamed variants (e.g., ListAccountRequest, ReqListAccountForward)

RULES:
- Do NOT invent APIs
- Do NOT guess fields not shown
- If something is not visible, omit it
- Return STRICT JSON ONLY

JSON FORMAT:
{
  "apis": [
    {
      "method": "POST",
      "path": "/example/path",
      "request_fields": ["field1", "field2"],
      "response_status": [200, 400]
    }
  ]
}
"""
    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_img}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0,
        "max_completion_tokens": 800,
        "response_format": {"type": "json_object"}
    }
    response = SESSION.post(
        GROQ_API_URL,
        headers=headers,
        json=payload,
        timeout=60
    )
    try:
        data = response.json()
    except Exception:
        raise RuntimeError(
            f"Groq did not return JSON.\nStatus: {response.status_code}\nBody:\n{response.text}"
        )

    if "choices" not in data:
        raise RuntimeError(
            f"Unexpected Groq response:\n{json.dumps(data, indent=2)}"
        )

    raw = data["choices"][0]["message"]["content"]
    # SAFETY: raw may already be dict OR string
    if isinstance(raw, str):
        extracted = json.loads(raw)
    else:
        extracted = raw

    # ✅ APPLY DEDUPLICATION HERE (CORRECT PLACE)
    if "apis" in extracted:
        extracted["apis"] = deduplicate_apis(extracted["apis"])
    return extracted


#####rag integration####

def enrich_with_rag(extracted):
    for api in extracted["apis"]:
        query = f"""
        API Path: {api['path']}
        Method: {api['method']}
        Provide request parameters and headers.
        """

        from rag.rag_engine import extract_params_from_text

        context_chunks = query_docs(query)

        # join retrieved chunks
        joined_context = "\n".join(context_chunks)

        # 🔥 deterministic parameter extraction
        params = extract_params_from_text(joined_context)

        api["rag_context"] = context_chunks
        api["request_fields"] = params

    return extracted


#############################
def extract_params_from_text(text):
    if not isinstance(text, str):
        return []

    # FIX broken parameter names
    text = re.sub(r"seq-\s*\n\s*no", "seq-no", text, flags=re.IGNORECASE)
    text = re.sub(r"channel-\s*\n\s*code", "channel-code", text, flags=re.IGNORECASE)

import re

def extract_params_from_text(text: str):
    """
    Extract request parameters from API documentation text.
    Works on sample requests like:
    DATA mobile=xxx&account-provider=1&device-id=xxx&seq-no=xxx&channel-code=xxx
    """

    params = set()

    # 1️⃣ Match query/body style params: key=value
    for match in re.findall(r"([a-zA-Z0-9_-]+)\s*=", text):
        # filter obvious non-params
        if len(match) >= 2:
            params.add(match)

    # 2️⃣ Remove garbage words (safety)
    blacklist = {
        "POST", "GET", "DATA", "https", "http", "server", "port"
    }

    params = [p for p in params if p.lower() not in blacklist]

    return sorted(params)


####new rag function####
def enrich_with_rag(extracted):
    for api in extracted["apis"]:
        query = f"""
        API Path: {api['path']}
        Method: {api['method']}
        Sample request
        """
        full_text = " ".join(api.get("_rag_context", []))
        api["request_fields"] = extract_params_from_text(full_text)

        rag_chunks = query_docs(query)

        api["_rag_context"] = rag_chunks

        relevant_texts = []

        for block in api["_rag_context"]:
            # Only take DATA blocks for ListAccount API
            if "/upi/la" in block.lower():
                relevant_texts.append(block)

        # fallback: if nothing matched, use everything
        if not relevant_texts:
            relevant_texts = api["_rag_context"]

        full_text = " ".join(relevant_texts)
        api["request_fields"] = extract_params_from_text(full_text)

    return extracted

########################rag enhancement function####
import re

def extract_params_from_text(text: str):
    if not isinstance(text, str):
        return []

    # 1️⃣ Fix broken words like seq-\nno
    text = re.sub(r"-\s*\n\s*", "-", text)

    # 2️⃣ Capture DATA blocks even if they span multiple lines
    data_blocks = re.findall(
        r"DATA\s+(.+?)(?:\n\s*\n|Response|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    params = set()

    for block in data_blocks:
        # join lines inside DATA
        block = block.replace("\n", " ")

        # split key=value pairs
        pairs = re.findall(r"([a-zA-Z0-9\-]+)\s*=", block)
        for p in pairs:
            params.add(p.strip().lower())

    return sorted(params)

# __________________________________
# yaml generation function
def generate_openapi_yaml(extracted):
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Generated API Specification",
            "version": "1.0.0"
        },
        "paths": {}
    }
    for api in extracted.get("apis", []):
        path = api["path"]
        method = api["method"].lower()
        if path not in spec["paths"]:
            spec["paths"][path] = {}
        responses = {}
        for status in api.get("response_status", []):
            responses[str(status)] = {
                "description": f"Response {status}"
            }
        operation = {
            "summary": f"{method.upper()} {path}",
            "responses": responses
        }
        # Add request body ONLY if fields exist
        if api.get("request_fields"):
            operation["requestBody"] = {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                field: {"type": "string"}
                                for field in api["request_fields"]
                            }
                        }
                    }
                }
            }
        spec["paths"][path][method] = operation
    return yaml.dump(spec, sort_keys=False)
##########################################################################
# #####################jmx xml
import xml.etree.ElementTree as ET
from xml.dom import minidom

def build_jmx_from_yaml(y):

    def ht(parent):
        return ET.SubElement(parent, "hashTree")

    jmeter = ET.Element(
        "jmeterTestPlan",
        attrib={"version": "1.2", "properties": "5.0", "jmeter": "5.6.3"}
    )
    root_ht = ht(jmeter)

    # ================= Test Plan =================
    tp = ET.SubElement(
        root_ht,
        "TestPlan",
        guiclass="TestPlanGui",
        testclass="TestPlan",
        testname="Generated Test Plan",
        enabled="true"
    )
    ET.SubElement(tp, "elementProp",
                  name="TestPlan.user_defined_variables",
                  elementType="Arguments")
    tp_ht = ht(root_ht)

    # ================= Thread Group =================
    tg = ET.SubElement(
        tp_ht,
        "ThreadGroup",
        guiclass="ThreadGroupGui",
        testclass="ThreadGroup",
        testname="Thread Group",
        enabled="true"
    )
    ET.SubElement(tg, "intProp", name="ThreadGroup.num_threads").text = "1"
    ET.SubElement(tg, "intProp", name="ThreadGroup.ramp_time").text = "1"
    ET.SubElement(tg, "stringProp", name="ThreadGroup.on_sample_error").text = "continue"

    lc = ET.SubElement(
        tg,
        "elementProp",
        name="ThreadGroup.main_controller",
        elementType="LoopController",
        guiclass="LoopControlPanel",
        testclass="LoopController"
    )
    ET.SubElement(lc, "stringProp", name="LoopController.loops").text = "1"
    ET.SubElement(lc, "boolProp", name="LoopController.continue_forever").text = "false"

    tg_ht = ht(tp_ht)

    # ================= JDBC DATASOURCE =================
    jdbc_ds = ET.SubElement(
        tg_ht,
        "JDBCDataSource",
        guiclass="TestBeanGUI",
        testclass="JDBCDataSource",
        testname="MySQL DB Connection",
        enabled="true"
    )
    for k, v in {
        "dataSource": "mysqlPool",
        "dbUrl": "${DB_URL}",
        "driver": "com.mysql.cj.jdbc.Driver",
        "username": "${DB_USER}",
        "password": "${DB_PASS}",
        "poolMax": "10",
        "timeout": "10000"
    }.items():
        ET.SubElement(jdbc_ds, "stringProp", name=k).text = v
    ET.SubElement(jdbc_ds, "boolProp", name="autocommit").text = "true"
    ht(tg_ht)

    COMMON_PARAMS = [
        "profile-id","channel-code","virtual-address",
        "device-id","institute_id","account-provider","seq-no"
    ]

    for path in y.get("paths", {}):

        sampler = ET.SubElement(
            tg_ht,
            "HTTPSamplerProxy",
            guiclass="HttpTestSampleGui",
            testclass="HTTPSamplerProxy",
            testname=f"POST {path}",
            enabled="true"
        )

        for k, v in {
            "HTTPSampler.domain": "localhost",
            "HTTPSampler.port": "8080",
            "HTTPSampler.protocol": "http",
            "HTTPSampler.path": path,
            "HTTPSampler.method": "POST"
        }.items():
            ET.SubElement(sampler, "stringProp", name=k).text = v

        ET.SubElement(sampler, "boolProp", name="HTTPSampler.postBodyRaw").text = "true"

        args = ET.SubElement(sampler, "elementProp",
                             name="HTTPsampler.Arguments",
                             elementType="Arguments")
        coll = ET.SubElement(args, "collectionProp", name="Arguments.arguments")
        body = ET.SubElement(coll, "elementProp", name="", elementType="HTTPArgument")
        ET.SubElement(body, "boolProp", name="HTTPArgument.always_encode").text = "false"
        ET.SubElement(body, "stringProp", name="Argument.value").text = "&".join(
            f"{p}=${{{p}}}" for p in COMMON_PARAMS
        )
        ET.SubElement(body, "stringProp", name="Argument.metadata").text = "="

        sampler_ht = ht(tg_ht)

        # ===== SEQ PREPROCESSOR =====
        pre = ET.SubElement(
            sampler_ht,
            "JSR223PreProcessor",
            guiclass="TestBeanGUI",
            testclass="JSR223PreProcessor",
            testname="Generate seq-no",
            enabled="true"
        )
        ET.SubElement(pre, "stringProp", name="scriptLanguage").text = "groovy"
        ET.SubElement(pre, "stringProp", name="script").text = """import java.util.Random;
String prefix = "JSFkernel";
String chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXTZabcdefghiklmnopqrstuvwxyz";
int string_length = 35 - prefix.length();
String randomstring = "";
Random randomGenerator = new Random();
for (int i = 0; i < string_length; i++) {
    int randomInt = randomGenerator.nextInt(chars.length());
    randomstring += chars.substring(randomInt, randomInt+1);
}
vars.put("seq-no", prefix + randomstring);"""
        ht(sampler_ht)

        # ===== SIMPLE CONTROLLER =====
        sc = ET.SubElement(
            sampler_ht,
            "GenericController",
            guiclass="LogicControllerGui",
            testclass="GenericController",
            testname="Post API Logic",
            enabled="true"
        )
        sc_ht = ht(sampler_ht)

        # ===== DEBUG =====
        debug = ET.SubElement(
            sc_ht,
            "DebugSampler",
            guiclass="TestBeanGUI",
            testclass="DebugSampler",
            testname="Debug Sampler",
            enabled="true"
        )
        ET.SubElement(debug, "boolProp", name="displayJMeterVariables").text = "true"
        ht(sc_ht)

        # ===== FETCH API RESPONSE =====
        fetch = ET.SubElement(
            sc_ht,
            "JSR223Sampler",
            guiclass="TestBeanGUI",
            testclass="JSR223Sampler",
            testname="Fetch API Response",
            enabled="true"
        )
        ET.SubElement(fetch, "stringProp", name="scriptLanguage").text = "groovy"
        ET.SubElement(fetch, "stringProp", name="script").text = """import org.apache.jmeter.samplers.SampleResult
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardOpenOption
import java.text.SimpleDateFormat
import java.util.Date
 
def dateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss+01:00") //(_get formattef date)

def time = new Date().format("yyyy-MM-dd HH-mm")

//def csvFilePath = "D:/JenkinsGlobalWorkspace/workspace/CBDC_LOAD/Current-Reports/API-CSV/report.csv"
def headers = "TestCaseID,APIResult,TestCaseName,combineResponse,DB_Responce,AssertionMsg,DBResult,DB_Request,StartTime,EndTime,combineRequest,ResponseCode"

def prev = prev

if (prev != null) {
    
    def requestData = "---# REQUEST BODY #---\n"+prev.getSamplerData()

//JIRA request data
    def JrequestData = prev.getSamplerData()
    def requestDatareplace = JrequestData.replace("\"", "\"\"")
      vars.put("JirarequestData", requestDatareplace)


    def responseData = "---# **RESPONSE BODY** #---\n" +prev.getResponseDataAsString()
    
   


def escapeQuotesOnly = { input -> input.replaceAll(/"/, '\\\\"') }
def result2 = escapeQuotesOnly(responseData)

  //  def threadName = prev.getThreadName()
   def threadGroupName = ctx.getThreadGroup().getName()
    //def threadGroupName = ctx.getThreadGroup().getName() ?: "Unknown_Thread_Group"
   // log.info("✅ Thread Group: " + threadGroupName)
   // vars.put("threadgrpname", threadName)
    vars.put("threadgrpname", threadGroupName)
    def httpReqName=prev.getSampleLabel()
    vars.put("Requestname", httpReqName)
    
    def reqheader="---------------# REQUEST HEADER #---------\n"+prev.getRequestHeaders()
    def Jreqheader=prev.getRequestHeaders()
     vars.put("Jirareqheader", Jreqheader)
    
    def respheader="---------------# RESPONSE HEADER #---------\n"+prev.getResponseHeaders()
    def Jrespheader=prev.getResponseHeaders()
    vars.put("Jirarespheader", Jrespheader)
     def time1= new Date(prev.getStartTime())
    String startTime = dateFormat.format(time1)  
     vars.put("StartTime", startTime)
     
    def time2= new Date(prev.getEndTime())
    String endTime = dateFormat.format(time2)
    vars.put("EndTime", endTime)
    def respcode = prev.getResponseCode()
     vars.put("RespCode", respcode)
    def result = prev.isSuccessful()
    
def assertionResults = prev.getAssertionResults()
def failedAssertions = []


if (assertionResults != null && assertionResults.size() > 0) {
    log.info("Number of assertions: " + assertionResults.size())
    for (int i = 0; i < assertionResults.size(); i++) {
        def assertionResult = assertionResults[i]
        if (assertionResult.isFailure()) {
            log.warn("Assertion " + (i + 1) + ": Failed")
            def failureMessage = assertionResult.getFailureMessage().replace("\"", "\"\"")
            log.warn("Failure message: " + failureMessage)
            failedAssertions.add(failureMessage)
        }
    }
    
    
    if (failedAssertions.size() > 0) {
        def allFailureMessages = failedAssertions.join(" | ") // Combine all failure messages
        vars.put("AsrResult", allFailureMessages)
    } else {
        vars.put("AsrResult", "Passed")
    }
    
} else {
    vars.put("AsrResult", "No assertion results")
    log.info("No assertion results found for the previous sampler.")
}

def AsrResult1=vars.get("AsrResult")

   def a = requestData.replace("\"", "\"\"")
       def b = result2.replace("\"", "\"\"")
       def noSpaces = b.replaceAll(",", "")
       def noSpaces2 = noSpaces.replaceAll(~/\s/,"")
       vars.put("Data", noSpaces2)
       
     def combineRequest = "${reqheader}|${a}"
  vars.put("CombineRequestHeaderBODY", combineRequest)
     
      def combineResponse = "${noSpaces}"
      

        vars.put("CombineResponceHeaderBODY", combineResponse)
     def resultStatus = result ? "PASSED" : "FAILED"
vars.put("ResultAPI", resultStatus)
//jiraresult
def jiraresultStatus = result ? "PASSED" : "FAILED"
vars.put("JIRAResultAPI", jiraresultStatus)
    
   def tcid= vars.get("TestCaseID")
 vars.put("TCid", tcid)
    
    def testscenario = vars.get("TestCaseName")
 vars.put("TestScenario", testscenario)  
    def csvEntry = "\"${tcid}\",${testscenario}\,\"${startTime}\",\"${endTime}\",\"${combineRequest}\",\"${combineResponse}\",\"${respcode}\",\"${resultStatus}\",\"${AsrResult1}\"\n"

    //Files.write(Paths.get(csvFilePath), csvEntry.getBytes(), StandardOpenOption.CREATE, StandardOpenOption.APPEND)
 
} """
        ht(sc_ht)
        
        
                # ===== JSR223 RESPONSE ASSERTION =====
        jsr_assert = ET.SubElement(
            sc_ht,
            "JSR223Assertion",
            guiclass="TestBeanGUI",
            testclass="JSR223Assertion",
            testname="JSR223 Response Assertion",
            enabled="true"
        )
        ET.SubElement(jsr_assert, "stringProp", name="scriptLanguage").text = "groovy"
        ET.SubElement(jsr_assert, "stringProp", name="script").text = """
import groovy.json.JsonSlurper

def response = prev.getResponseDataAsString()
def json = new JsonSlurper().parseText(response)

assert json.Success == true
assert json.ActCode == "0"
assert json.message == "Transaction Successful"
assert json.Response == "Transaction Successful"
assert json.response == "0"
assert json.MobileAppData["original-txn-message"] == "Transaction initiated"
assert json.MobileAppData["original-txn-response-code"]=="92"
"""
        ht(sc_ht)


        # ===== DB VERIFICATION =====
        dbv = ET.SubElement(
            sc_ht,
            "JDBCSampler",
            guiclass="TestBeanGUI",
            testclass="JDBCSampler",
            testname="DB Verification",
            enabled="true"
        )
        ET.SubElement(dbv, "stringProp", name="dataSource").text = "mysqlPool"
        ET.SubElement(dbv, "stringProp", name="queryType").text = "Select Statement"
        ET.SubElement(dbv, "stringProp", name="query").text = (
            "select type,status,irc,itc,display_message "
            "from upi_tranlog where txn_id='${seq-no}'"
        )
        ht(sc_ht)
        
        
                # ===== POST PROCESSOR =====
        post = ET.SubElement(
            sc_ht,
            "JSR223PostProcessor",
            guiclass="TestBeanGUI",
            testclass="JSR223PostProcessor",
            testname="Post Processor",
            enabled="true"
        )
        ET.SubElement(post, "stringProp", name="scriptLanguage").text = "groovy"
        ET.SubElement(post, "stringProp", name="script").text = """
vars.put("http_status", prev.getResponseCode())
vars.put("response_time", prev.getTime().toString())
"""
        ht(sc_ht)


        # ===== SAVE CSV =====
        save_csv = ET.SubElement(
            sc_ht,
            "JSR223Sampler",
            guiclass="TestBeanGUI",
            testclass="JSR223Sampler",
            testname="Save API & DB to CSV",
            enabled="true"
        )
        ET.SubElement(save_csv, "stringProp", name="scriptLanguage").text = "groovy"
        ET.SubElement(save_csv, "stringProp", name="script").text = """import org.apache.jmeter.samplers.SampleResult
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardOpenOption
import java.text.SimpleDateFormat
import java.util.Date
def csvFilePath = "C:/JenkinsGlobalWorkspace/workspace/UPI_JANA/Current-Reports/API-CSV/report.csv"
def headers = "ThreadGrpName,TestCaseName,StartTime,EndTime,combineRequest,combineRequest,ResponseCode,APIResult,AssertionMsg,DB_Request,DB_Responce,DBResult,DBAssertionMsg,OverallApiResult"
// Get the previous SampleResult
def prev = prev
// Check if the SampleResult is not null
if (prev != null) { 
def dThreadGrpName= vars.get("threadgrpname")
def dTestCaseName = vars.get("Requestname")
def dStartTime= vars.get("StartTime")
def dEndTime = vars.get("EndTime")
def dcombineRequest= vars.get("CombineRequestHeaderBODY")
//def dcombineResponse = vars.get("CombineResponceHeaderBODY")
def drespcode= vars.get("RespCode")
def httpreqresult = vars.get("ResultAPI")
def dAPIResult = vars.get("ResultAPI")
//vars.put("Apiresult",dAPIResult)
String dAssertionMsg= vars.get("AsrResult")
// Removethe spaces 
String actualResponse = vars.get("CombineResponceHeaderBODY")
String NoSpaces = actualResponse.replaceAll("\\s+", "")
// NoQuotes = NoSpaces.
//String NoQuotes = NoSpaces.replaceAll(" ", "\\\" \\\"")
//vars.put("ActualResponseNoSpaces", NoSpaces)
//def dThreadGrpName = prev.getThreadName()
//replace(" ", "\\\" \\\"")
String resp = vars.get("CombineResponceHeaderBODY")
String responseBodyEscaped = resp.replaceAll("\"", "\\\\\"");

vars.put("ActualResponseNoSpaces", responseBodyEscaped)
def dcombineResponse = vars.get("Data")


//DB Related Operation
//DB Related Operation
def dDB_Request=prev.getSamplerData().replace("\"", "\"\"").replace(",", "")
vars.put("jdbcRequestData", dDB_Request)
String dbRequest = vars.get("jdbcRequestData")
String DBReqNoSpaces = dbRequest.replaceAll("\\s+", "")
vars.put("DBReqNoSpaces1", DBReqNoSpaces)
def newdbreq = vars.get("DBReqNoSpaces1")
def responseData = prev.getResponseDataAsString()
//def modifiedResponseData = responseData.replaceAll("\\s", "_")
def dDB_Responce = responseData.replaceAll("[\\n]", "|")
vars.put("jdbcResponcetData", dDB_Responce)
String dbResponse = vars.get("jdbcResponcetData")
String DBResponseNoSpaces = dbResponse.replaceAll("\\s+", "")
vars.put("DBResponseNoSpaces1", DBResponseNoSpaces)
def newdbresp = vars.get("DBResponseNoSpaces1")

//DB Result Assertion
def assertionResults = prev.getAssertionResults()
// Check if there are assertion results
if (assertionResults != null && assertionResults.size() > 0) {
    log.info("Number of assertions: " + assertionResults.size())
    // Iterate through each assertion result
    for (int i = 0; i < assertionResults.size(); i++) {
        def assertionResult = assertionResults[i]
        // Check if the assertion passed or failed
        if (!assertionResult.isFailure()) {
            log.info("Assertion " + (i + 1) + ": Passed")
            def asrDBResult="PASSED"
            vars.put("AsrDBResult", asrDBResult)
            def dbasertionmsg = "All assertions are passed" 
            vars.put("DBassertionmsg",dbasertionmsg)
        } else {
                def asrDBResult="FAILED"
                 vars.put("AsrDBResult", asrDBResult)
            log.warn("Assertion " + (i + 1) + ": Failed")
            def asrDBmsg = assertionResult.getFailureMessage().replace("\"", "\"\"") 
            log.warn("Failure message: " + asrDBmsg)
           vars.put("DBassertionmsg",asrDBmsg)
           
        }
    }
} else {
    def asrDBResult="PASSED"
    vars.put("AsrDBResult", asrDBResult)
     def asrDBmsg = "No assertion results found"
     vars.put("DBassertionmsg", asrDBmsg)
    log.info("No assertion results found for the previous sampler.")
}
def AsrDBResult1=vars.get("AsrDBResult")
def dbAssertionMsg = vars.get("DBassertionmsg")
   
    if (!Files.exists(Paths.get(csvFilePath))) {
    Files.write(Paths.get(csvFilePath), (headers + "\n").getBytes(), StandardOpenOption.CREATE)
}
if (dAPIResult == "PASSED" && AsrDBResult1 == "PASSED" ){
    log.info("dAPI",dAPIResult)
def Overallresult = "PASSED"
vars.put("Result1", Overallresult)
    }
    else{
def Overallresult = "FAILED"
vars.put("Result1", Overallresult)
        }
def Result2 = vars.get("Result1")    
    
   
def csvEntry = "\"${dThreadGrpName} : ${dTestCaseName}\",\"${dTestCaseName}\",\"${dStartTime}\",\"${dEndTime}\",\"${dcombineRequest}\",\"${dcombineResponse}\",\"${drespcode}\",\"${httpreqresult}\",\"${dAssertionMsg}\",\"${dDB_Request}\",\"${dDB_Responce}\",\"${AsrDBResult1}\",\"${dbAssertionMsg}\",\"${Result2}\"\n" 

    Files.write(Paths.get(csvFilePath), csvEntry.getBytes(), StandardOpenOption.CREATE, StandardOpenOption.APPEND)
 
}   """
        ht(sc_ht)

    # ===== VIEW RESULTS TREE =====
    ET.SubElement(
        tp_ht,
        "ResultCollector",
        guiclass="ViewResultsFullVisualizer",
        testclass="ResultCollector",
        testname="View Results Tree",
        enabled="true"
    )
    ht(tp_ht)

    xml = ET.tostring(jmeter, "utf-8")
    return minidom.parseString(xml).toprettyxml(indent="  ")

# ===============================
# STREAMLIT UI
# ===============================

# ===============================
# STREAMLIT UI
# ===============================

st.set_page_config(page_title="Sequence Diagram → API Extractor", layout="wide")
st.title("Sequence Diagram → API Extraction (VISION LOCKED)")

groq_key = st.text_input("Groq API Key", type="password")
uploaded = st.file_uploader("Upload Sequence Diagram", type=["png", "jpg", "jpeg"])

if uploaded:
    st.image(uploaded, use_column_width=True)

if uploaded and groq_key and st.button("Extract APIs"):
    with st.spinner("Analyzing diagram with Vision Model..."):
        img_bytes = compress_image(uploaded)
        extracted = extract_apis_from_diagram(groq_key, img_bytes)
        extracted = enrich_with_rag(extracted)
        st.subheader("RAG Enrichment (Per API – Debug)")
        for api in extracted.get("apis", []):
            st.markdown(f"### {api['method']} {api['path']}")
            st.text(str(api.get("_rag_context", ""))[:1500])


        
        # 🔹 Build a query for RAG
        if extracted.get("apis"):
            rag_query = " ; ".join(
                f"{api['method']} {api['path']}"
                for api in extracted["apis"]
            )
            extracted["_rag_context"] = query_docs(rag_query)
        else:
            extracted["_rag_context"] = ""

        # ✅ NOW it is safe to display
        st.subheader("RAG Retrieved Context (DEBUG)")
        st.text(extracted["_rag_context"][:1500])

        st.subheader("Extracted APIs (Structured)")
        st.json(extracted)
        
        openapi_yaml = generate_openapi_yaml(extracted)
        st.subheader("Generated OpenAPI YAML")
        st.code(openapi_yaml, language="yaml")
        yaml_spec = yaml.safe_load(openapi_yaml)
        jmx = build_jmx_from_yaml(yaml_spec)
        st.subheader("Generated JMX")
        st.download_button(
            "Download JMX",
            jmx,
            file_name="plan.jmx",
            mime="application/xml"
        )
        
        ##############################################
        if not extracted.get("apis"):
            st.error("❌ No APIs detected. Diagram may be unclear.")
            st.stop()
        else:
            st.success(f"✅ {len(extracted['apis'])} API(s) extracted successfully")

st.write("App loaded successfully.")
      