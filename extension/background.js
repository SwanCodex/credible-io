chrome.action.onClicked.addListener((tab) => {
  console.log("Credible.io extension clicked!");

  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    files: ["content.js"]
  });
});

chrome.runtime.onMessage.addListener((message, sender) => {
  if (message.type === "PAGE_TEXT") {
    console.log("Sending text to /repair endpoint...");

    fetch("http://127.0.0.1:8000/repair", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(message.payload)
    })
      .then(res => res.json())
      .then(data => {
        console.log("Repair response:", data);

        chrome.tabs.sendMessage(sender.tab.id, {
          type: "SHOW_REPAIR",
          payload: data
        });
      })
      .catch(err => {
        console.error("Repair request failed:", err);
      });
  }

  if (message.type === "ANALYZE_BIAS") {
    console.log("Sending text to /analyze-bias endpoint...");

    fetch("http://127.0.0.1:8000/analyze-bias", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(message.payload)
    })
      .then(res => res.json())
      .then(data => {
        console.log("Bias analysis response:", data);

        chrome.tabs.sendMessage(sender.tab.id, {
          type: "SHOW_BIAS_ANALYSIS",
          payload: data
        });
      })
      .catch(err => {
        console.error("Bias analysis request failed:", err);
      });
  }

  if (message.type === "GET_XAI_INFO") {
    console.log("Sending text to /xai-info endpoint...");

    fetch("http://127.0.0.1:8000/xai-info", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(message.payload)
    })
      .then(res => res.json())
      .then(data => {
        console.log("XAI info response:", data);

        chrome.tabs.sendMessage(sender.tab.id, {
          type: "SHOW_XAI_INFO",
          payload: data
        });
      })
      .catch(err => {
        console.error("XAI info request failed:", err);
      });
  }

  if (message.type === "ANALYZE_KNOWLEDGE_GAPS") {
    console.log("Sending text to /knowledge-gaps endpoint...");

    fetch("http://127.0.0.1:8000/knowledge-gaps", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(message.payload)
    })
      .then(res => res.json())
      .then(data => {
        console.log("Knowledge gaps response:", data);

        chrome.tabs.sendMessage(sender.tab.id, {
          type: "SHOW_KNOWLEDGE_GAPS",
          payload: data
        });
      })
      .catch(err => {
        console.error("Knowledge gaps request failed:", err);
      });
  }
});