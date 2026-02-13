"""
Update n8n Workflows for LIVE DATA
Updates existing workflows to fetch data from Python backend instead of static data
"""

import requests
import json

N8N_BASE_URL = "http://localhost:5678/api/v1"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI4ZDc2ODY1MS1hZjYzLTQzNDctOWNiOC1lZjA1MTllZWM2NjgiLCJpc3MiOiJuOG4iLCJhdWQiOiJwdWJsaWMtYXBpIiwiaWF0IjoxNzcwNjY1ODc4LCJleHAiOjE3NzMyMDE2MDB9.k9Vzu_RQGpVKNYhqMhXdxfonmKvmbeyhQdzJbhAyFcU"

headers = {
    "X-N8N-API-KEY": API_KEY,
    "Content-Type": "application/json"
}

def delete_workflow(workflow_id):
    """Delete a workflow"""
    try:
        response = requests.delete(
            f"{N8N_BASE_URL}/workflows/{workflow_id}",
            headers=headers
        )
        if response.status_code in [200, 204]:
            print(f"[+] Deleted workflow: {workflow_id}")
            return True
        else:
            print(f"[-] Failed to delete {workflow_id}: {response.text}")
            return False
    except Exception as e:
        print(f"[-] Error: {e}")
        return False

def create_workflow(workflow_data):
    """Create a workflow"""
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
        print(f"[-] Failed: {response.text}")
        return None

# ============================================
# UPDATED WORKFLOWS WITH LIVE DATA
# ============================================

# Workflow 1: Disaster Trigger Agent (Updated for full-response endpoint)
disaster_trigger_live = {
    "name": "Disaster Trigger Agent (LIVE)",
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
                "responseMode": "lastNode"
            },
            "webhookId": "disaster-trigger-live"
        },
        {
            "id": "http_full",
            "name": "Full Disaster Response",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "position": [500, 300],
            "parameters": {
                "url": "http://localhost:5000/full-response",
                "method": "POST",
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": "={{ JSON.stringify({lat: $json.lat, lng: $json.lng, patients: $json.patients, disaster_type: $json.disaster_type || 'FIRE', source: 'osm'}) }}"
            }
        },
        {
            "id": "respond1",
            "name": "Return Response",
            "type": "n8n-nodes-base.respondToWebhook",
            "typeVersion": 1.1,
            "position": [750, 300],
            "parameters": {
                "respondWith": "json",
                "responseBody": "={{ $json }}"
            }
        }
    ],
    "connections": {
        "Disaster Webhook": {
            "main": [[{"node": "Full Disaster Response", "type": "main", "index": 0}]]
        },
        "Full Disaster Response": {
            "main": [[{"node": "Return Response", "type": "main", "index": 0}]]
        }
    },
    "settings": {"executionOrder": "v1"}
}

# Workflow 2: Hospital Agent (LIVE - fetches from Python backend)
hospital_agent_live = {
    "name": "Hospital Agent (LIVE)",
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
                "responseMode": "lastNode"
            },
            "webhookId": "hospital-data-live"
        },
        {
            "id": "http_optimize",
            "name": "Optimize Allocation (LIVE)",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "position": [500, 300],
            "parameters": {
                "url": "http://localhost:5000/optimize",
                "method": "POST",
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": "={{ JSON.stringify({lat: $json.lat, lng: $json.lng, patients: $json.patients, source: 'osm'}) }}"
            }
        },
        {
            "id": "http_medicine",
            "name": "Get Medicine Requirements",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "position": [750, 300],
            "parameters": {
                "url": "http://localhost:5000/medicine-requirements",
                "method": "POST",
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": "={{ JSON.stringify({disaster_type: 'FIRE', patients: $('Hospital Webhook').first().json.patients}) }}"
            }
        },
        {
            "id": "merge1",
            "name": "Merge Data",
            "type": "n8n-nodes-base.code",
            "typeVersion": 2,
            "position": [950, 300],
            "parameters": {
                "jsCode": """
// Merge allocation and medicine data
const allocation = $('Optimize Allocation (LIVE)').first().json;
const medicines = $('Get Medicine Requirements').first().json;

return [{
    json: {
        source: 'hospital',
        allocation: allocation.allocation,
        remaining: allocation.remaining,
        data_source: allocation.data_source,
        medicine_requirements: {
            priority: medicines.priority,
            critical_items: medicines.critical_items,
            equipment: medicines.equipment
        }
    }
}];
"""
            }
        },
        {
            "id": "respond2",
            "name": "Respond",
            "type": "n8n-nodes-base.respondToWebhook",
            "typeVersion": 1.1,
            "position": [1150, 300],
            "parameters": {
                "respondWith": "json",
                "responseBody": "={{ $json }}"
            }
        }
    ],
    "connections": {
        "Hospital Webhook": {
            "main": [[{"node": "Optimize Allocation (LIVE)", "type": "main", "index": 0}]]
        },
        "Optimize Allocation (LIVE)": {
            "main": [[{"node": "Get Medicine Requirements", "type": "main", "index": 0}]]
        },
        "Get Medicine Requirements": {
            "main": [[{"node": "Merge Data", "type": "main", "index": 0}]]
        },
        "Merge Data": {
            "main": [[{"node": "Respond", "type": "main", "index": 0}]]
        }
    },
    "settings": {"executionOrder": "v1"}
}

# Workflow 3: Emergency Agent (LIVE)
emergency_agent_live = {
    "name": "Emergency Agent (LIVE)",
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
                "responseMode": "lastNode"
            },
            "webhookId": "ambulance-data-live"
        },
        {
            "id": "http_ambulance",
            "name": "Dispatch Ambulances (LIVE)",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "position": [500, 300],
            "parameters": {
                "url": "http://localhost:5000/ambulance-dispatch",
                "method": "POST",
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": "={{ JSON.stringify({lat: $json.lat, lng: $json.lng, source: 'osm'}) }}"
            }
        },
        {
            "id": "code_format",
            "name": "Format Response",
            "type": "n8n-nodes-base.code",
            "typeVersion": 2,
            "position": [750, 300],
            "parameters": {
                "jsCode": """
const dispatch = $input.first().json;
return [{
    json: {
        source: 'emergency',
        dispatch: dispatch.dispatch,
        total_ambulances: dispatch.total_ambulances,
        data_source: dispatch.data_source
    }
}];
"""
            }
        },
        {
            "id": "respond3",
            "name": "Respond",
            "type": "n8n-nodes-base.respondToWebhook",
            "typeVersion": 1.1,
            "position": [950, 300],
            "parameters": {
                "respondWith": "json",
                "responseBody": "={{ $json }}"
            }
        }
    ],
    "connections": {
        "Ambulance Webhook": {
            "main": [[{"node": "Dispatch Ambulances (LIVE)", "type": "main", "index": 0}]]
        },
        "Dispatch Ambulances (LIVE)": {
            "main": [[{"node": "Format Response", "type": "main", "index": 0}]]
        },
        "Format Response": {
            "main": [[{"node": "Respond", "type": "main", "index": 0}]]
        }
    },
    "settings": {"executionOrder": "v1"}
}

# Workflow 4: Medicine Alert Agent (NEW)
medicine_alert_agent = {
    "name": "Medicine Alert Agent",
    "nodes": [
        {
            "id": "webhook4",
            "name": "Medicine Webhook",
            "type": "n8n-nodes-base.webhook",
            "typeVersion": 2,
            "position": [250, 300],
            "parameters": {
                "path": "medicine-alert",
                "httpMethod": "POST",
                "responseMode": "lastNode"
            },
            "webhookId": "medicine-alert-id"
        },
        {
            "id": "http_meds",
            "name": "Get Medicine Requirements",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "position": [500, 300],
            "parameters": {
                "url": "http://localhost:5000/medicine-requirements",
                "method": "POST",
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": "={{ JSON.stringify({disaster_type: $json.disaster_type || 'FIRE', patients: $json.patients || 6}) }}"
            }
        },
        {
            "id": "http_checklist",
            "name": "Get Hospital Checklist",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "position": [750, 300],
            "parameters": {
                "url": "http://localhost:5000/hospital-checklist",
                "method": "POST",
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": "={{ JSON.stringify({disaster_type: $json.disaster_type || 'FIRE', patients: $json.patients || 6, hospital: $json.hospital || 'City Hospital'}) }}"
            }
        },
        {
            "id": "respond4",
            "name": "Respond",
            "type": "n8n-nodes-base.respondToWebhook",
            "typeVersion": 1.1,
            "position": [950, 300],
            "parameters": {
                "respondWith": "json",
                "responseBody": "={{ $json }}"
            }
        }
    ],
    "connections": {
        "Medicine Webhook": {
            "main": [[{"node": "Get Medicine Requirements", "type": "main", "index": 0}]]
        },
        "Get Medicine Requirements": {
            "main": [[{"node": "Get Hospital Checklist", "type": "main", "index": 0}]]
        },
        "Get Hospital Checklist": {
            "main": [[{"node": "Respond", "type": "main", "index": 0}]]
        }
    },
    "settings": {"executionOrder": "v1"}
}

def main():
    print("=" * 60)
    print("Updating n8n Workflows for LIVE DATA")
    print("=" * 60)
    
    # Get existing workflow IDs
    response = requests.get(f"{N8N_BASE_URL}/workflows", headers=headers)
    existing = response.json().get("data", [])
    
    # Find and delete old workflows that we're replacing
    old_names = ["Disaster Trigger Agent", "Hospital Agent", "Emergency Agent"]
    for wf in existing:
        if wf["name"] in old_names:
            print(f"[!] Deleting old workflow: {wf['name']}")
            delete_workflow(wf["id"])
    
    # Create new LIVE workflows
    workflows = [
        disaster_trigger_live,
        hospital_agent_live,
        emergency_agent_live,
        medicine_alert_agent
    ]
    
    created_ids = []
    for wf in workflows:
        result = create_workflow(wf)
        if result:
            created_ids.append(result.get("id"))
    
    print("\n" + "=" * 60)
    print("LIVE Workflows Created!")
    print("=" * 60)
    print("\nWebhook URLs:")
    print("  - Full Response:    http://localhost:5678/webhook/disaster-trigger")
    print("  - Hospital (LIVE):  http://localhost:5678/webhook/hospital-data")
    print("  - Ambulance (LIVE): http://localhost:5678/webhook/ambulance-data")
    print("  - Medicine Alert:   http://localhost:5678/webhook/medicine-alert")
    print("\nNOTE: Activate workflows in n8n UI (toggle ON)")
    print("=" * 60)

if __name__ == "__main__":
    main()
