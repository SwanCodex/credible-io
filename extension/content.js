(() => {
  // 1. Extract page text and send to background - create menu for user choice
  const pageText = document.body.innerText;

  // Create analysis menu
  showAnalysisMenu(pageText);

  // 2. Listen for all types of results
  chrome.runtime.onMessage.addListener((message) => {
    if (message.type === "SHOW_REPAIR") {
      showRepairPanel(message.payload);
    }
    if (message.type === "SHOW_BIAS_ANALYSIS") {
      showBiasAnalysisPanel(message.payload);
    }
    if (message.type === "SHOW_XAI_INFO") {
      showXAIPanel(message.payload);
    }
    if (message.type === "SHOW_KNOWLEDGE_GAPS") {
      showKnowledgeGapsPanel(message.payload);
    }
  });

  // 3. Create analysis menu
  function showAnalysisMenu(text) {
    // Remove existing menu if present
    const existing = document.getElementById("credible-analysis-menu");
    if (existing) existing.remove();

    const menu = document.createElement("div");
    menu.id = "credible-analysis-menu";

    menu.innerHTML = `
      <h3>Credible.io Analysis Menu</h3>
      <p>Select analysis type:</p>
      
      <button id="credible-repair-btn">Misinformation Repair</button>
      <button id="credible-bias-btn">Bias & Perspective Analysis</button>
      <button id="credible-xai-btn">XAI & Intuitive Info</button>
      <button id="credible-gaps-btn">Knowledge Gap Mapping</button>
      <button id="credible-menu-close-btn">Close</button>
    `;

    // Styling
    menu.style.position = "fixed";
    menu.style.top = "20px";
    menu.style.right = "20px";
    menu.style.width = "300px";
    menu.style.background = "#ffffff";
    menu.style.border = "2px solid #333";
    menu.style.padding = "15px";
    menu.style.zIndex = "999999";
    menu.style.fontSize = "14px";
    menu.style.boxShadow = "0 4px 12px rgba(0,0,0,0.3)";

    document.body.appendChild(menu);

    // Button handlers
    document.getElementById("credible-repair-btn").onclick = () => {
      menu.remove();
      chrome.runtime.sendMessage({
        type: "PAGE_TEXT",
        payload: {
          content: text,
          source_url: window.location.href
        }
      });
    };

    document.getElementById("credible-bias-btn").onclick = () => {
      menu.remove();
      chrome.runtime.sendMessage({
        type: "ANALYZE_BIAS",
        payload: {
          content: text,
          source_url: window.location.href
        }
      });
    };

    document.getElementById("credible-xai-btn").onclick = () => {
      menu.remove();
      chrome.runtime.sendMessage({
        type: "GET_XAI_INFO",
        payload: {
          content: text,
          source_url: window.location.href
        }
      });
    };

    document.getElementById("credible-gaps-btn").onclick = () => {
      menu.remove();
      chrome.runtime.sendMessage({
        type: "ANALYZE_KNOWLEDGE_GAPS",
        payload: {
          content: text,
          source_url: window.location.href
        }
      });
    };

    document.getElementById("credible-menu-close-btn").onclick = () => {
      menu.remove();
    };
  }

  // 4. Create repair overlay UI
  function showRepairPanel(data) {
    // Remove existing panel if present
    const existing = document.getElementById("credible-repair-panel");
    if (existing) existing.remove();

    const panel = document.createElement("div");
    panel.id = "credible-repair-panel";

    panel.innerHTML = `
      <h3>Credible.io – Live Misinformation Repair</h3>

      <strong>Original:</strong>
      <p>${truncate(data.original_text)}</p>

      <strong>Repaired:</strong>
      <p>${truncate(data.repaired_text)}</p>

      <strong>Why this was repaired:</strong>
      <p>${data.repair_explanation}</p>

      <button id="credible-close-btn">Close</button>
    `;

    // Styling (inline to avoid CSS injection issues)
    panel.style.position = "fixed";
    panel.style.top = "20px";
    panel.style.right = "20px";
    panel.style.width = "360px";
    panel.style.maxHeight = "80vh";
    panel.style.overflowY = "auto";
    panel.style.background = "#ffffff";
    panel.style.border = "2px solid #333";
    panel.style.padding = "12px";
    panel.style.zIndex = "999999";
    panel.style.fontSize = "14px";
    panel.style.boxShadow = "0 4px 12px rgba(0,0,0,0.3)";

    document.body.appendChild(panel);

    document.getElementById("credible-close-btn").onclick = () => {
      panel.remove();
    };
  }

  // 5. Create bias analysis panel
  function showBiasAnalysisPanel(data) {
    const existing = document.getElementById("credible-bias-panel");
    if (existing) existing.remove();

    const panel = document.createElement("div");
    panel.id = "credible-bias-panel";

    panel.innerHTML = `
      <h3>Bias & Perspective Analysis</h3>
      
      <strong>Bias Score:</strong> ${data.bias_score}/100<br>
      <strong>Bias Level:</strong> ${data.bias_level}<br>
      <strong>Objectivity:</strong> ${data.objectivity_rating}/100<br><br>
      
      <strong>Assessment:</strong>
      <p>${data.overall_assessment}</p>
      
      <strong>Flags:</strong>
      <ul>${data.bias_flags.map(f => `<li>${f}</li>`).join("") || "<li>None</li>"}</ul>
      
      <strong>Recommendations:</strong>
      <ul>${data.recommendations.map(r => `<li>${truncate(r, 150)}</li>`).join("") || "<li>Content appears balanced</li>"}</ul>
      
      <button id="credible-bias-close-btn">Close</button>
    `;

    stylePanel(panel);
    document.body.appendChild(panel);

    document.getElementById("credible-bias-close-btn").onclick = () => {
      panel.remove();
    };
  }

  // 6. Create XAI info panel
  function showXAIPanel(data) {
    const existing = document.getElementById("credible-xai-panel");
    if (existing) existing.remove();

    const panel = document.createElement("div");
    panel.id = "credible-xai-panel";

    panel.innerHTML = `
      <h3>XAI & Intuitive Info</h3>
      
      <strong>Confidence:</strong> ${data.overall_confidence}%<br>
      <strong>Level:</strong> ${data.confidence_level}<br><br>
      
      <strong>Explanation:</strong>
      <p>${data.confidence_explanation}</p>
      
      <strong>Analysis Methods:</strong>
      <ul>${data.analysis_methods_used.slice(0, 3).map(m => `<li>${truncate(m, 100)}</li>`).join("")}</ul>
      
      <strong>Limitations:</strong>
      <ul>${data.known_limitations.slice(0, 3).map(l => `<li>${truncate(l, 100)}</li>`).join("")}</ul>
      
      <strong>User Guidance:</strong>
      <ul>${data.user_guidance.map(g => `<li>${truncate(g, 120)}</li>`).join("")}</ul>
      
      <button id="credible-xai-close-btn">Close</button>
    `;

    stylePanel(panel);
    document.body.appendChild(panel);

    document.getElementById("credible-xai-close-btn").onclick = () => {
      panel.remove();
    };
  }

  // 7. Create knowledge gaps panel
  function showKnowledgeGapsPanel(data) {
    const existing = document.getElementById("credible-gaps-panel");
    if (existing) existing.remove();

    const panel = document.createElement("div");
    panel.id = "credible-gaps-panel";

    panel.innerHTML = `
      <h3>Knowledge Gap Mapping</h3>
      
      <strong>Completeness:</strong> ${data.completeness_score}/100<br>
      <strong>Assessment:</strong> ${data.completeness_assessment}<br><br>
      
      <strong>Knowledge Gaps:</strong>
      <ul>${data.knowledge_gaps_identified.map(g => `<li>${g}</li>`).join("") || "<li>None detected</li>"}</ul>
      
      <strong>Prerequisites:</strong>
      <ul>${data.prerequisite_knowledge.map(p => `<li>${truncate(p, 100)}</li>`).join("") || "<li>None identified</li>"}</ul>
      
      <strong>Depth:</strong> ${data.depth_analysis.level} (${data.depth_analysis.score}/100)<br><br>
      
      <strong>Recommendations:</strong>
      <ul>${data.actionable_recommendations.map(r => `<li>${truncate(r, 120)}</li>`).join("") || "<li>Content appears comprehensive</li>"}</ul>
      
      <button id="credible-gaps-close-btn">Close</button>
    `;

    stylePanel(panel);
    document.body.appendChild(panel);

    document.getElementById("credible-gaps-close-btn").onclick = () => {
      panel.remove();
    };
  }

  // 8. Helper to style panels
  function stylePanel(panel) {
    panel.style.position = "fixed";
    panel.style.top = "20px";
    panel.style.right = "20px";
    panel.style.width = "360px";
    panel.style.maxHeight = "80vh";
    panel.style.overflowY = "auto";
    panel.style.background = "#ffffff";
    panel.style.border = "2px solid #333";
    panel.style.padding = "12px";
    panel.style.zIndex = "999999";
    panel.style.fontSize = "14px";
    panel.style.boxShadow = "0 4px 12px rgba(0,0,0,0.3)";
  }

  // 9. Helper to keep UI readable
    if (!text) return "";
    return text.length > limit
      ? text.slice(0, limit) + "..."
      : text;
  }
})();
