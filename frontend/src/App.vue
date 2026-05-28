<template>
  <div class="app-shell">
    <header class="hero">
      <div>
        <p class="eyebrow">Agent Deepfake Face Detector</p>
        <h1>基于双智能体的 Deepfake 人脸图像检测系统</h1>
        <p class="hero-copy">
          特征提取智能体负责视觉证据生成，决策智能体负责策略选择、语义辅助判断与结构化证据链输出。
        </p>
      </div>
      <div class="hero-meta">
        <span>三值输出</span>
        <span>策略选择</span>
        <span>可审计证据链</span>
        <span
          :class="llmReady ? 'meta-badge meta-badge--ok' : 'meta-badge meta-badge--off'"
          :title="llmReady ? `LLM 策略选择已启用（${llmModel}）` : '未配置 DASHSCOPE_API_KEY，回退到规则策略'"
        >
          {{ llmReady ? `LLM 已启用 · ${llmModel}` : 'LLM 未就绪（规则回退）' }}
        </span>
      </div>
    </header>

    <div v-if="globalError" class="global-error-bar">
      <span>{{ globalError }}</span>
      <button @click="globalError = ''">关闭</button>
    </div>

    <main class="dashboard">
      <section class="left-column">
        <UploadPanel :uploading="uploading" @submit="handleUpload" />
        <TaskHistory
          :tasks="tasks"
          :active-task-id="selectedTaskId"
          @select="selectTask"
          @refresh="refreshTasks"
        />
      </section>
      <section class="right-column">
        <ResultPanel :status="selectedStatus" :result="selectedResult" />
        <EvidenceTimeline :entries="selectedResult?.evidence_chain || []" />
      </section>
    </main>
  </div>
</template>

<script setup>
import { onMounted, ref } from "vue";
import { createTask, getSystemStatus, getTaskResult, getTaskStatus, listTasks } from "./api/client";
import EvidenceTimeline from "./components/EvidenceTimeline.vue";
import ResultPanel from "./components/ResultPanel.vue";
import TaskHistory from "./components/TaskHistory.vue";
import UploadPanel from "./components/UploadPanel.vue";

const tasks = ref([]);
const selectedTaskId = ref("");
const selectedStatus = ref(null);
const selectedResult = ref(null);
const uploading = ref(false);
const globalError = ref("");
const llmReady = ref(false);
const llmModel = ref("");

async function refreshTasks() {
  tasks.value = await listTasks(30);
  const taskFromQuery = new URLSearchParams(window.location.search).get("task");
  const preferredTask = taskFromQuery || selectedTaskId.value;
  if (preferredTask && tasks.value.some((item) => item.task_id === preferredTask)) {
    await selectTask(preferredTask, false);
    return;
  }
  if (!selectedTaskId.value && tasks.value.length) {
    await selectTask(tasks.value[0].task_id, false);
  }
}

async function handleUpload(file) {
  try {
    uploading.value = true;
    const response = await createTask(file);
    selectedTaskId.value = response.task_id;
    updateTaskQuery(response.task_id);
    selectedStatus.value = await getTaskStatus(response.task_id);
    selectedResult.value = null;
    await refreshTasks();
    pollUntilDone(response.task_id);
  } catch (error) {
    globalError.value = error.message;
  } finally {
    uploading.value = false;
  }
}

async function selectTask(taskId, syncQuery = true) {
  selectedTaskId.value = taskId;
  if (syncQuery) updateTaskQuery(taskId);
  selectedStatus.value = await getTaskStatus(taskId);
  if (selectedStatus.value.status === "completed") {
    selectedResult.value = await getTaskResult(taskId);
  } else if (selectedStatus.value.status === "failed") {
    selectedResult.value = null;
    globalError.value = selectedStatus.value.error_message || "任务执行失败";
  } else {
    selectedResult.value = null;
    pollUntilDone(taskId);
  }
}

async function pollUntilDone(taskId) {
  const maxAttempts = 60;
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    const status = await getTaskStatus(taskId);
    if (taskId !== selectedTaskId.value) return;
    selectedStatus.value = status;
    if (status.status === "completed") {
      selectedResult.value = await getTaskResult(taskId);
      await refreshTasks();
      return;
    }
    if (status.status === "failed") {
      globalError.value = status.error_message || "任务执行失败";
      await refreshTasks();
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 1200));
  }
}

function updateTaskQuery(taskId) {
  const url = new URL(window.location.href);
  url.searchParams.set("task", taskId);
  window.history.replaceState({}, "", url);
}

onMounted(async () => {
  try {
    const status = await getSystemStatus();
    llmReady.value = status?.llm?.effective_status === "ready";
    llmModel.value = status?.llm?.text_model || "";
  } catch {
    // ignore
  }
  refreshTasks().catch((error) => {
    console.error(error);
  });
});
</script>
