"""
n8n Workflow Setup Script
Creates all 4 agent workflows via n8n REST API
"""

import requests
import json

N8N_BASE_URL = "http://localhost:5678/api/v1"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI4ZDc2ODY1MS1hZjYzLTQzNDctOWNiOC1lZjA1MTllZWM2NjgiLCJpc3MiOiJuOG4iLCJhdWQiOiJwdWJsaWMtYXBpIiwiaWF0IjoxNzcwNjY1ODc4LCJleHAiOjE3NzMyMDE2MDB9.k9Vzu_RQGpVKNYhqMhXdxfonmKvmbeyhQdzJbhAyFcU"

headers = {
    "X-N8N-API-KEY": API_KEY,
    "Content-Type": "application/json"
}

def create_workflow(workflow_data):
    """Create a workflow via n8n API"""
    response = requests.post(
        f"{N8N_BASE_URL}/workflows",
        headers=headers,
        json=workflow_data
    )
    if response.status_code in [200, 201]:
        result = response.json()
        print(f"[+] Created workflow: {workflow_data['name']} (ID: {result.get('id', 'N/A')})")
        return result
    else:
        print(f"[-] Failed to create {workflow_data['name']}: {response.text}")
        return None

def activate_workflow(workflow_id):
    """Activate a workflow"""
    response = requests.patch(
        f"{N8N_BASE_URL}/workflows/{workflow_id}",
        headers=headers,
        json={"active": True}
    )
    if response.status_code == 200:
        print(f"    [+] Activated workflow ID: {workflow_id}")
    else:
        print(f"    [-] Failed to activate: {response.text}")

# Workflow 1: Disaster Trigger Agent
disaster_trigger_workflow = {
    "name": "Disaster Trigger Agent",
    "nodes": [
        {
            "id": "webhook1",
            "name": "Disaster Webhook",
            "type": "n8n-nodes-base.webhook",
            "typeVersion": 2,
            "position": [250, 300],
            "parameters": {
                "path": "disaster-trigger",
                "httpMethod": "POST",
                "responseMode": "onReceived"
            },
            "webhookId": "disaster-trigger-id"
        },
        {
            "id": "set1", 
            "name": "Add Metadata",
            "type": "n8n-nodes-base.set",
            "typeVersion": 3.4,
            "position": [450, 300],
            "parameters": {
                "mode": "manual",
                "duplicateItem": False,
                "assignments": {
                    "assignments": [
                        {"id": "sev1", "name": "severity", "value": "HIGH", "type": "string"},
                        {"id": "type1", "name": "type", "value": "FIRE", "type": "string"},
                        {"id": "lat1", "name": "lat", "value": "={{ $json.lat }}", "type": "number"},
                        {"id": "lng1", "name": "lng", "value": "={{ $json.lng }}", "type": "number"},
                        {"id": "pat1", "name": "patients", "value": "={{ $json.patients }}", "type": "number"}
                    ]
                }
            }
        },
        {
            "id": "http1",
            "name": "Call Hospital Agent",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "position": [700, 200],
            "parameters": {
                "url": "http://localhost:5678/webhook/hospital-data",
                "method": "POST",
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": "={{ JSON.stringify({lat: $json.lat, lng: $json.lng, patients: $json.patients}) }}"
            }
        },
        {
            "id": "http2",
            "name": "Call Emergency Agent", 
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "position": [700, 400],
            "parameters": {
                "url": "http://localhost:5678/webhook/ambulance-data",
                "method": "POST",
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": "={{ JSON.stringify({lat: $json.lat, lng: $json.lng, patients: $json.patients}) }}"
            }
        }
    ],
    "connections": {
        "Disaster Webhook": {
            "main": [[{"node": "Add Metadata", "type": "main", "index": 0}]]
        },
        "Add Metadata": {
            "main": [[
                {"node": "Call Hospital Agent", "type": "main", "index": 0},
                {"node": "Call Emergency Agent", "type": "main", "index": 0}
            ]]
        }
    },
    "settings": {"executionOrder": "v1"}
}

# Workflow 2: Hospital Agent
hospital_agent_workflow = {
    "name": "Hospital Agent",
    "nodes": [
        {
            "id": "webhook2",
            "name": "Hospital Webhook",
            "type": "n8n-nodes-base.webhook",
            "typeVersion": 2,
            "position": [250, 300],
            "parameters": {
                "path": "hospital-data",
                "httpMethod": "POST",
                "responseMode": "onReceived"
            },
            "webhookId": "hospital-data-id"
        },
        {
            "id": "code1",
            "name": "Hospital Database",
            "type": "n8n-nodes-base.code",
            "typeVersion": 2,
            "position": [450, 300],
            "parameters": {
                "jsCode": """
// Hospital Database with bed availability
const hospitals = [
    { Hospital: "A", Beds: 3, lat: 22.722, lng: 88.481 },
    { Hospital: "B", Beds: 5, lat: 22.726, lng: 88.490 },
    { Hospital: "C", Beds: 2, lat: 22.718, lng: 88.478 }
];

// Get disaster info from webhook
const disasterInfo = $input.first().json;

return [{
    json: {
        patients: disasterInfo.patients,
        lat: disasterInfo.lat,
        lng: disasterInfo.lng,
        hospitals: hospitals
    }
}];
"""
            }
        },
        {
            "id": "http3",
            "name": "Call Optimization",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "position": [650, 300],
            "parameters": {
                "url": "http://localhost:5000/optimize",
                "method": "POST",
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": "={{ JSON.stringify($json) }}"
            }
        },
        {
            "id": "http4",
            "name": "Send to Government",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "position": [850, 300],
            "parameters": {
                "url": "http://localhost:5678/webhook/decision",
                "method": "POST",
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": "={{ JSON.stringify({source: 'hospital', allocation: $json.allocation, remaining: $json.remaining}) }}"
            }
        }
    ],
    "connections": {
        "Hospital Webhook": {
            "main": [[{"node": "Hospital Database", "type": "main", "index": 0}]]
        },
        "Hospital Database": {
            "main": [[{"node": "Call Optimization", "type": "main", "index": 0}]]
        },
        "Call Optimization": {
            "main": [[{"node": "Send to Government", "type": "main", "index": 0}]]
        }
    },
    "settings": {"executionOrder": "v1"}
}

# Workflow 3: Emergency Agent
emergency_agent_workflow = {
    "name": "Emergency Agent",
    "nodes": [
        {
            "id": "webhook3",
            "name": "Ambulance Webhook",
            "type": "n8n-nodes-base.webhook",
            "typeVersion": 2,
            "position": [250, 300],
            "parameters": {
                "path": "ambulance-data",
                "httpMethod": "POST",
                "responseMode": "onReceived"
            },
            "webhookId": "ambulance-data-id"
        },
        {
            "id": "code2",
            "name": "Ambulance Fleet",
            "type": "n8n-nodes-base.code",
            "typeVersion": 2,
            "position": [450, 300],
            "parameters": {
                "jsCode": """
// Ambulance Fleet Database
const ambulances = [
    { id: "AMB-001", lat: 22.720, lng: 88.480, status: "available" },
    { id: "AMB-002", lat: 22.724, lng: 88.487, status: "available" },
    { id: "AMB-003", lat: 22.718, lng: 88.492, status: "available" }
];

const disasterInfo = $input.first().json;

// Simple distance calculation
function getDistance(lat1, lng1, lat2, lng2) {
    return Math.sqrt(Math.pow(lat2-lat1, 2) + Math.pow(lng2-lng1, 2)) * 111; // approx km
}

// Assign ambulances by distance
const dispatch = ambulances.map(amb => ({
    ambulance_id: amb.id,
    distance_km: getDistance(disasterInfo.lat, disasterInfo.lng, amb.lat, amb.lng).toFixed(2),
    eta_minutes: (getDistance(disasterInfo.lat, disasterInfo.lng, amb.lat, amb.lng) * 3).toFixed(1)
})).sort((a,b) => a.distance_km - b.distance_km);

return [{ json: { source: 'emergency', dispatch: dispatch } }];
"""
            }
        },
        {
            "id": "http5",
            "name": "Send to Government",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "position": [650, 300],
            "parameters": {
                "url": "http://localhost:5678/webhook/decision",
                "method": "POST",
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": "={{ JSON.stringify($json) }}"
            }
        }
    ],
    "connections": {
        "Ambulance Webhook": {
            "main": [[{"node": "Ambulance Fleet", "type": "main", "index": 0}]]
        },
        "Ambulance Fleet": {
            "main": [[{"node": "Send to Government", "type": "main", "index": 0}]]
        }
    },
    "settings": {"executionOrder": "v1"}
}

# Workflow 4: Government Decision Agent
government_agent_workflow = {
    "name": "Government Decision Agent",
    "nodes": [
        {
            "id": "webhook4",
            "name": "Decision Webhook",
            "type": "n8n-nodes-base.webhook",
            "typeVersion": 2,
            "position": [250, 300],
            "parameters": {
                "path": "decision",
                "httpMethod": "POST",
                "responseMode": "lastNode"
            },
            "webhookId": "decision-id"
        },
        {
            "id": "code3",
            "name": "Process Decision",
            "type": "n8n-nodes-base.code",
            "typeVersion": 2,
            "position": [450, 300],
            "parameters": {
                "jsCode": """
// Government Decision Processing
const data = $input.first().json;

let response = {
    status: "DECISION_MADE",
    timestamp: new Date().toISOString(),
    source: data.source || "unknown"
};

if (data.source === 'hospital') {
    response.hospital_allocation = data.allocation;
    response.remaining_patients = data.remaining;
    response.decision = data.remaining > 0 ? 
        "NEED_ADDITIONAL_RESOURCES" : "ALL_PATIENTS_ALLOCATED";
} else if (data.source === 'emergency') {
    response.ambulance_dispatch = data.dispatch;
    response.decision = "AMBULANCES_DISPATCHED";
}

return [{ json: response }];
"""
            }
        },
        {
            "id": "respond1",
            "name": "Respond",
            "type": "n8n-nodes-base.respondToWebhook",
            "typeVersion": 1.1,
            "position": [650, 300],
            "parameters": {
                "respondWith": "json",
                "responseBody": "={{ $json }}"
            }
        }
    ],
    "connections": {
        "Decision Webhook": {
            "main": [[{"node": "Process Decision", "type": "main", "index": 0}]]
        },
        "Process Decision": {
            "main": [[{"node": "Respond", "type": "main", "index": 0}]]
        }
    },
    "settings": {"executionOrder": "v1"}
}

def main():
    print("=" * 50)
    print("n8n Multi-Agent Workflow Setup")
    print("=" * 50)
    
    workflows = [
        disaster_trigger_workflow,
        hospital_agent_workflow,
        emergency_agent_workflow,
        government_agent_workflow
    ]
    
    created_ids = []
    
    for workflow in workflows:
        result = create_workflow(workflow)
        if result:
            created_ids.append(result.get('id'))
    
    print("\n" + "=" * 50)
    print("Activating Workflows...")
    print("=" * 50)
    
    for wf_id in created_ids:
        if wf_id:
            activate_workflow(wf_id)
    
    print("\n" + "=" * 50)
    print("Setup Complete!")
    print("=" * 50)
    print("\nWebhook URLs:")
    print("  - Disaster Trigger: http://localhost:5678/webhook/disaster-trigger")
    print("  - Hospital Data:    http://localhost:5678/webhook/hospital-data")
    print("  - Ambulance Data:   http://localhost:5678/webhook/ambulance-data")
    print("  - Decision:         http://localhost:5678/webhook/decision")

if __name__ == "__main__":
    main()
