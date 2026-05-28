<template>
  <section class="panel">
    <div class="panel-head">
      <div>
        <p class="eyebrow">证据链时间线</p>
        <h2>结构化审计记录</h2>
      </div>
      <StatusBadge :text="entries.length ? 'completed' : 'idle'" />
    </div>

    <div v-if="!entries.length" class="empty-state">
      任务开始后，这里会展示策略选择、语义触发、规则命中和最终结论。
    </div>

    <div v-else class="timeline">
      <article
        v-for="entry in entries"
        :key="`${entry.step}-${entry.timestamp}`"
        class="timeline-item"
        :class="{ 'timeline-item--llm': hasThinkingChain(entry) }"
      >
        <div class="timeline-dot" :class="{ 'timeline-dot--llm': hasThinkingChain(entry) }" />
        <div class="timeline-card">
          <div class="timeline-head">
            <strong>{{ entry.step }}</strong>
            <div class="timeline-head-right">
              <span v-if="hasThinkingChain(entry)" class="llm-badge">LLM 思维链</span>
              <span>{{ formatTime(entry.timestamp) }}</span>
            </div>
          </div>
          <p>{{ entry.summary }}</p>

          <!-- LLM 思维决策链路专属展示区 -->
          <template v-if="hasThinkingChain(entry)">
            <div class="thinking-chain-block">
              <div class="thinking-chain-head" @click="toggleThinking(entry.step + entry.timestamp)">
                <span class="thinking-icon">🧠</span>
                <span>大模型推理过程</span>
                <span class="thinking-toggle">{{ expandedKeys.has(entry.step + entry.timestamp) ? '▲ 收起' : '▼ 展开' }}</span>
              </div>
              <div v-if="expandedKeys.has(entry.step + entry.timestamp)" class="thinking-chain-body">
                <pre>{{ entry.details.thinking_chain }}</pre>
              </div>
            </div>
            <!-- 其余详情（去掉 thinking_chain 字段） -->
            <details class="raw-details">
              <summary>原始字段</summary>
              <pre>{{ formatDetailsWithoutChain(entry.details) }}</pre>
            </details>
          </template>

          <template v-else>
            <pre>{{ formatDetails(entry.details) }}</pre>
          </template>
        </div>
      </article>
    </div>
  </section>
</template>

<script setup>
import { ref } from "vue";
import StatusBadge from "./StatusBadge.vue";

defineProps({
  entries: {
    type: Array,
    default: () => []
  }
});

const expandedKeys = ref(new Set());

function hasThinkingChain(entry) {
  return Boolean(entry.details && entry.details.thinking_chain);
}

function toggleThinking(key) {
  const next = new Set(expandedKeys.value);
  if (next.has(key)) {
    next.delete(key);
  } else {
    next.add(key);
  }
  expandedKeys.value = next;
}

function formatTime(timestamp) {
  return new Date(timestamp).toLocaleString("zh-CN");
}

function formatDetails(details) {
  return JSON.stringify(details, null, 2);
}

function formatDetailsWithoutChain(details) {
  if (!details) return "";
  const { thinking_chain, ...rest } = details;
  return JSON.stringify(rest, null, 2);
}
</script>

<style scoped>
.timeline-item--llm > .timeline-card {
  border-left: 3px solid #7c6af7;
}

.timeline-dot--llm {
  background: #7c6af7 !important;
  box-shadow: 0 0 0 4px rgba(124, 106, 247, 0.2);
}

.timeline-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
}

.timeline-head-right {
  display: flex;
  align-items: center;
  gap: 8px;
}

.llm-badge {
  font-size: 11px;
  font-weight: 600;
  background: linear-gradient(90deg, #7c6af7, #a78bfa);
  color: #fff;
  padding: 2px 8px;
  border-radius: 99px;
  letter-spacing: 0.03em;
}

.thinking-chain-block {
  margin-top: 10px;
  border: 1px solid rgba(124, 106, 247, 0.35);
  border-radius: 8px;
  overflow: hidden;
}

.thinking-chain-head {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: rgba(124, 106, 247, 0.08);
  cursor: pointer;
  font-size: 13px;
  font-weight: 600;
  user-select: none;
}

.thinking-chain-head:hover {
  background: rgba(124, 106, 247, 0.15);
}

.thinking-icon {
  font-size: 15px;
}

.thinking-toggle {
  margin-left: auto;
  font-size: 12px;
  color: #7c6af7;
}

.thinking-chain-body pre {
  margin: 0;
  padding: 12px 14px;
  font-size: 12px;
  line-height: 1.65;
  white-space: pre-wrap;
  word-break: break-word;
  background: rgba(124, 106, 247, 0.04);
  border-top: 1px solid rgba(124, 106, 247, 0.15);
}

.raw-details {
  margin-top: 6px;
  font-size: 12px;
}

.raw-details summary {
  cursor: pointer;
  color: #888;
  padding: 2px 0;
}

.raw-details pre {
  margin-top: 4px;
  font-size: 11px;
}
</style>
