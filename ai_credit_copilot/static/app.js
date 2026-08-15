(function () {
  "use strict";

  var state = {
    chatHistory: [],
    modelName: "qwen3.5",
    toastTimer: null
  };

  var defaultEvidence = [
    { label: "近期还款波动", impact: "up", value: "近 6 个月存在 2 次逾期信号", confidence: 0.86 },
    { label: "债务负担变化", impact: "up", value: "聚合后的债务收入比处于规则关注区间", confidence: 0.79 },
    { label: "现金流稳定性", impact: "down", value: "已观察到连续的收入流入模式", confidence: 0.73 }
  ];

  function byId(id) {
    return document.getElementById(id);
  }

  function setText(id, value) {
    byId(id).textContent = value == null ? "" : String(value);
  }

  function createElement(tag, className, text) {
    var element = document.createElement(tag);
    if (className) {
      element.className = className;
    }
    if (text != null) {
      element.textContent = text;
    }
    return element;
  }

  function percent(value) {
    return Math.round(Number(value || 0) * 100) + "%";
  }

  function showToast(message, isError) {
    var toast = byId("toast");
    toast.textContent = message;
    toast.classList.toggle("is-error", Boolean(isError));
    toast.classList.add("is-visible");
    window.clearTimeout(state.toastTimer);
    state.toastTimer = window.setTimeout(function () {
      toast.classList.remove("is-visible");
    }, 4200);
  }

  function setBusy(button, busy, busyText) {
    if (!button.dataset.defaultText) {
      button.dataset.defaultText = button.textContent;
    }
    button.disabled = busy;
    button.textContent = busy ? busyText : button.dataset.defaultText;
  }

  function fetchJson(url, options) {
    return fetch(url, options).then(function (response) {
      return response.json().catch(function () { return {}; }).then(function (body) {
        if (!response.ok) {
          var error = new Error(body.message || body.error || "服务请求失败");
          error.code = body.error;
          throw error;
        }
        return body;
      });
    });
  }

  function riskBand(score) {
    if (score >= 0.7) {
      return { name: "高关注", css: "is-high" };
    }
    if (score >= 0.4) {
      return { name: "中关注", css: "is-medium" };
    }
    return { name: "低关注", css: "is-low" };
  }

  function updateScore() {
    setText("scoreValue", percent(byId("scoreRange").value));
  }

  function createEvidenceRow(item) {
    var row = createElement("div", "evidence-row");
    var label = document.createElement("input");
    label.type = "text";
    label.className = "evidence-label";
    label.maxLength = 80;
    label.placeholder = "证据标签";
    label.value = item.label || "";
    label.setAttribute("aria-label", "证据标签");

    var impact = document.createElement("select");
    impact.className = "evidence-impact";
    impact.setAttribute("aria-label", "信号方向");
    [
      ["up", "上行"],
      ["down", "缓释"],
      ["neutral", "中性"]
    ].forEach(function (optionValue) {
      var option = document.createElement("option");
      option.value = optionValue[0];
      option.textContent = optionValue[1];
      impact.appendChild(option);
    });
    impact.value = item.impact || "neutral";

    var value = document.createElement("input");
    value.type = "text";
    value.className = "evidence-value";
    value.maxLength = 220;
    value.placeholder = "聚合描述";
    value.value = item.value || "";
    value.setAttribute("aria-label", "聚合描述");

    var remove = document.createElement("button");
    remove.type = "button";
    remove.className = "remove-evidence";
    remove.textContent = "×";
    remove.title = "移除证据";
    remove.setAttribute("aria-label", "移除证据");
    remove.addEventListener("click", function () {
      row.remove();
    });

    row.appendChild(label);
    row.appendChild(impact);
    row.appendChild(value);
    row.appendChild(remove);
    return row;
  }

  function renderEvidence(items) {
    var list = byId("evidenceList");
    list.innerHTML = "";
    items.forEach(function (item) {
      list.appendChild(createEvidenceRow(item));
    });
  }

  function readEvidence() {
    return Array.prototype.slice.call(document.querySelectorAll(".evidence-row")).map(function (row) {
      return {
        label: row.querySelector(".evidence-label").value.trim(),
        impact: row.querySelector(".evidence-impact").value,
        value: row.querySelector(".evidence-value").value.trim(),
        confidence: 0.75
      };
    }).filter(function (item) {
      return item.label || item.value;
    });
  }

  function buildCase() {
    return {
      case_id: byId("caseId").value.trim(),
      model_score: Number(byId("scoreRange").value),
      model_name: "CrediFusion",
      model_version: byId("modelVersion").value.trim(),
      evidence: readEvidence(),
      federation: {
        party_count: Number(byId("partyCount").value),
        alignment_rate: Number(byId("alignmentRate").value),
        feature_coverage: Number(byId("coverageRate").value)
      },
      data_quality: byId("dataQuality").value.trim()
    };
  }

  function populateCase(data) {
    var item = data.case || data;
    byId("caseId").value = item.case_id || "CASE-2026-0081";
    byId("scoreRange").value = Number(item.model_score || 0.72).toFixed(2);
    byId("partyCount").value = String((item.federation && item.federation.party_count) || 2);
    byId("alignmentRate").value = (item.federation && item.federation.alignment_rate) || 0;
    byId("coverageRate").value = (item.federation && item.federation.feature_coverage) || 0;
    byId("modelVersion").value = item.model_version || "unspecified";
    byId("dataQuality").value = item.data_quality || "not_provided";
    renderEvidence(item.evidence && item.evidence.length ? item.evidence : defaultEvidence);
    updateScore();
  }

  function clearList(id, values) {
    var list = byId(id);
    list.innerHTML = "";
    if (!values || !values.length) {
      var empty = createElement("li", "", "未提供");
      list.appendChild(empty);
      return;
    }
    values.forEach(function (value) {
      list.appendChild(createElement("li", "", typeof value === "string" ? value : JSON.stringify(value)));
    });
  }

  function renderFactors(factors) {
    var list = byId("factorList");
    list.innerHTML = "";
    if (!factors || !factors.length) {
      list.appendChild(createElement("p", "summary-copy", "模型未返回结构化关键审查信号。"));
      setText("factorCount", "");
      return;
    }
    setText("factorCount", factors.length + " 项");
    factors.forEach(function (factor) {
      var signal = factor.signal === "down" ? "down" : factor.signal === "up" ? "up" : "neutral";
      var row = createElement("div", "factor-row impact-" + signal);
      var tagNames = { up: "上行", down: "缓释", neutral: "中性" };
      row.appendChild(createElement("span", "impact-tag", tagNames[signal]));
      var main = createElement("div", "factor-main");
      main.appendChild(createElement("strong", "", factor.factor || "待核验信号"));
      if (factor.rationale) {
        main.appendChild(createElement("p", "", factor.rationale));
      }
      if (factor.review_action) {
        main.appendChild(createElement("p", "factor-action", factor.review_action));
      }
      row.appendChild(main);
      list.appendChild(row);
    });
  }

  function renderAnalysis(data) {
    var caseData = data.case || {};
    var model = caseData.federated_model || {};
    var structured = data.analysis.structured || {};
    var score = Number(model.risk_probability || 0);
    var band = riskBand(score);

    byId("analysisEmpty").hidden = true;
    byId("analysisResult").hidden = false;
    setText("resultScore", percent(score));
    setText("riskBand", band.name);
    var scale = document.querySelector(".band-scale");
    scale.className = "band-scale " + band.css;
    setText("scoreFootnote", "来自 " + (model.name || "联邦模型") + " 的待复核模型信号");
    setText("reviewMeta", "Qwen · " + (data.analysis.model || state.modelName));
    setText("analysisSummary", structured.executive_summary || data.analysis.content);
    renderFactors(structured.risk_factors || []);
    clearList("mitigantList", structured.mitigants || []);
    clearList("questionList", structured.review_questions || []);
    clearList("limitationList", structured.limitations || ["模型未返回结构化限制说明，需由人工复核。"]);
    setText("complianceNote", structured.compliance_note || "仅供人工复核，不能作为自动化决策依据。");
    setText("rawOutput", data.analysis.content);
    setText("privacySummary", "已剔除 " + ((caseData.privacy && caseData.privacy.discarded_field_count) || 0) + " 个非白名单字段");
    setText("caseState", "已分析");
  }

  function appendChat(role, content) {
    var log = byId("chatLog");
    var placeholder = log.querySelector(".chat-placeholder");
    if (placeholder) {
      placeholder.remove();
    }
    var message = createElement("div", "chat-message " + role, content);
    log.appendChild(message);
    log.scrollTop = log.scrollHeight;
  }

  function resetReview() {
    byId("analysisEmpty").hidden = false;
    byId("analysisResult").hidden = true;
    setText("reviewMeta", "等待模型响应");
    setText("caseState", "草稿");
    byId("chatLog").innerHTML = "";
    byId("chatLog").appendChild(createElement("p", "chat-placeholder", "等待审查员问题"));
    state.chatHistory = [];
  }

  function loadDemo() {
    return fetchJson("/api/demo-case").then(function (data) {
      populateCase(data);
      resetReview();
      showToast("已载入脱敏样例案件。", false);
    }).catch(function () {
      populateCase({ evidence: defaultEvidence });
      resetReview();
      showToast("样例文件暂不可用，已载入本地表单内容。", true);
    });
  }

  function inspectHealth() {
    return fetchJson("/api/health").then(function (health) {
      state.modelName = health.model || "qwen3.5";
      setText("qwenModelName", state.modelName);
      setText("connectionText", health.configured ? (state.modelName + " 已配置") : (state.modelName + " 等待密钥"));
      byId("connectionStatus").className = "connection " + (health.configured ? "is-ready" : "is-error");
    }).catch(function () {
      setText("connectionText", "接口不可用");
      byId("connectionStatus").className = "connection is-error";
    });
  }

  function onAnalyze(event) {
    event.preventDefault();
    var button = byId("analyzeButton");
    var payload = { case: buildCase() };
    setBusy(button, true, "正在生成...");
    fetchJson("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).then(function (data) {
      renderAnalysis(data);
      showToast("已生成待人工复核的审查意见。", false);
    }).catch(function (error) {
      showToast(error.code === "qwen_not_configured" ? "Qwen 接口未配置，请设置 QWEN_API_KEY。" : error.message, true);
    }).finally(function () {
      setBusy(button, false, "");
    });
  }

  function onChat(event) {
    event.preventDefault();
    var questionInput = byId("chatQuestion");
    var question = questionInput.value.trim();
    if (!question) {
      return;
    }
    var button = byId("sendButton");
    var history = state.chatHistory.slice(-6);
    appendChat("user", question);
    state.chatHistory.push({ role: "user", content: question });
    questionInput.value = "";
    setBusy(button, true, "...");
    fetchJson("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ case: buildCase(), question: question, history: history })
    }).then(function (data) {
      appendChat("assistant", data.answer);
      state.chatHistory.push({ role: "assistant", content: data.answer });
      if (data.privacy) {
        setText("privacySummary", "已剔除 " + data.privacy.discarded_field_count + " 个非白名单字段");
      }
    }).catch(function (error) {
      appendChat("assistant", "未获得模型响应：" + error.message);
    }).finally(function () {
      setBusy(button, false, "");
    });
  }

  function init() {
    renderEvidence(defaultEvidence);
    updateScore();
    byId("scoreRange").addEventListener("input", updateScore);
    byId("addEvidenceButton").addEventListener("click", function () {
      byId("evidenceList").appendChild(createEvidenceRow({ impact: "neutral" }));
    });
    byId("caseForm").addEventListener("submit", onAnalyze);
    byId("chatForm").addEventListener("submit", onChat);
    byId("loadDemoButton").addEventListener("click", loadDemo);
    inspectHealth();
  }

  document.addEventListener("DOMContentLoaded", init);
}());
