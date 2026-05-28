<template>
  <section class="panel result-panel">
    <div class="panel-head">
      <div>
        <p class="eyebrow">结果详情</p>
        <h2>双智能体检测结果</h2>
      </div>
      <StatusBadge :text="result ? result.label : status?.status || 'idle'" />
    </div>

    <div v-if="status && status.status !== 'completed' && !result" class="status-card">
      <p>当前阶段：{{ status.stage }}</p>
      <div class="progress-track">
        <span class="progress-fill" :style="{ width: `${Math.round((status.progress || 0) * 100)}%` }" />
      </div>
      <p>执行智能体：{{ status.current_agent || "等待调度" }}</p>
    </div>

    <div v-else-if="result" class="result-body">
      <div class="result-summary">
        <div class="hero-score">
          <span class="score-label">图像级结论</span>
          <strong>{{ result.label }}</strong>
          <span>融合分数 {{ Number(result.fusion_score).toFixed(2) }}</span>
          <span>置信度 {{ Number(result.confidence).toFixed(2) }}</span>
        </div>
        <div class="summary-texts">
          <p><strong>视觉摘要：</strong>{{ result.visual_summary }}</p>
          <p><strong>语义摘要：</strong>{{ result.semantic_summary }}</p>
          <p v-if="result.review_reason"><strong>复核原因：</strong>{{ result.review_reason }}</p>
          <p v-if="thresholdSource"><strong>阈值来源：</strong>{{ thresholdSource }}</p>
        </div>
      </div>

      <!-- LLM 策略思维决策链路 -->
      <div v-if="strategyThinkingChain" class="thinking-chain-panel">
        <div class="thinking-chain-header" @click="showThinkingChain = !showThinkingChain">
          <span class="llm-dot" />
          <span>决策智能体推理链路（通义千问 LLM 策略选择）</span>
          <span class="tc-toggle">{{ showThinkingChain ? '▲ 收起' : '▼ 展开查看' }}</span>
        </div>
        <div v-if="showThinkingChain" class="thinking-chain-content">
          <pre>{{ strategyThinkingChain }}</pre>
        </div>
      </div>

      <div class="chips model-chip-row" v-if="modelEntries.length">
        <span v-for="entry in modelEntries" :key="entry.key" class="chip">
          {{ entry.label }}：{{ entry.value }}
        </span>
      </div>

      <div class="artifact-grid">
        <article v-if="annotatedArtifact" class="artifact-card wide">
          <div class="artifact-head" style="margin-bottom:10px">
            <h3 style="margin:0">原图标注</h3>
            <span class="face-count-badge" :class="result.faces.length ? 'badge-detected' : 'badge-none'">
              {{ result.faces.length ? `检测到 ${result.faces.length} 张人脸` : '未检测到人脸' }}
            </span>
          </div>
          <div class="annotated-img-wrap">
            <img :src="annotatedArtifact.url" alt="原图标注" @error="annotatedError = true" />
            <div v-if="annotatedError" class="img-error-placeholder">图像加载失败</div>
          </div>
        </article>
        <article
          v-for="face in result.faces"
          :key="face.face_id"
          class="artifact-card"
        >
          <div class="artifact-head">
            <h3>{{ face.face_id }}</h3>
            <StatusBadge :text="face.deepfake_score >= 0.7 ? '疑似伪造' : face.deepfake_score <= 0.3 ? '疑似真实' : '需人工复核'" />
          </div>
          <div class="artifact-pair">
            <img
              v-for="artifact in face.artifacts"
              :key="artifact.name"
              :src="artifact.url"
              :alt="artifact.description || artifact.kind"
            />
          </div>
          <div class="metric-grid">
            <span>视觉分数 {{ Number(face.deepfake_score).toFixed(2) }}</span>
            <span>质量分数 {{ Number(face.quality_score).toFixed(2) }}</span>
            <span>DCT 分数 {{ Number(face.dct_score).toFixed(2) }}</span>
            <span>一致性 {{ Number(face.augmentation_consistency_score).toFixed(2) }}</span>
          </div>
          <p v-if="face.public_figure_candidates?.length" class="face-note">
            公共人物候选：{{ face.public_figure_candidates.join("、") }}
          </p>
        </article>
      </div>

      <div class="chips">
        <span v-for="flag in result.semantic_flags" :key="flag.code" class="chip">
          {{ flag.label }} ({{ Number(flag.confidence).toFixed(2) }})
        </span>
      </div>
    </div>

    <div v-else class="empty-state">上传图像后，这里会展示原图、人脸对齐图、热力图和最终结论。</div>
  </section>
</template>

<script setup>
import { computed, ref } from "vue";
import StatusBadge from "./StatusBadge.vue";

const props = defineProps({
  status: {
    type: Object,
    default: null
  },
  result: {
    type: Object,
    default: null
  }
});

const showThinkingChain = ref(false);
const annotatedError = ref(false);

// 从 evidence_chain 的 strategy_selection 条目中提取 LLM 思维链
const strategyThinkingChain = computed(() => {
  const entry = props.result?.evidence_chain?.find((e) => e.step === "strategy_selection");
  return entry?.details?.thinking_chain || null;
});

const annotatedArtifact = computed(() =>
  props.result?.artifacts?.find((item) => item.kind === "annotated_original")
);

const thresholdSource = computed(() => props.result?.model_versions?.threshold_source || "");

const modelLabelMap = {
  detector: "检测后端",
  detector_provider: "检测 Provider",
  visual: "视觉后端",
  visual_model_name: "视觉模型",
  fusion: "融合后端",
  fusion_model_name: "融合模型",
  semantic: "语义后端",
  threshold_source: "阈值来源",
  true_threshold: "真图阈值",
  fake_threshold: "伪造阈值"
};

const modelEntries = computed(() => {
  const versions = props.result?.model_versions || {};
  return Object.entries(versions)
    .filter(([, value]) => value !== "" && !String(value).includes("_path"))
    .map(([key, value]) => ({
      key,
      label: modelLabelMap[key] || key,
      value
    }));
});
</script>

<style scoped>
.annotated-img-wrap {
  position: relative;
  width: 100%;
  min-height: 200px;
  background: #f0f4f2;
  border-radius: 14px;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}

.annotated-img-wrap img {
  width: 100%;
  height: auto;
  min-height: 200px;
  max-height: 600px;
  object-fit: contain;
  border-radius: 14px;
  display: block;
}

.img-error-placeholder {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #888;
  font-size: 14px;
  background: #f5f5f5;
}

.face-count-badge {
  font-size: 12px;
  font-weight: 600;
  padding: 4px 10px;
  border-radius: 999px;
}

.badge-detected {
  background: rgba(17, 145, 124, 0.12);
  color: #0b6f5f;
}

.badge-none {
  background: rgba(221, 141, 50, 0.14);
  color: #b06a10;
}

.thinking-chain-panel {
  margin: 12px 0;
  border: 1px solid rgba(124, 106, 247, 0.4);
  border-radius: 10px;
  overflow: hidden;
}

.thinking-chain-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 14px;
  background: rgba(124, 106, 247, 0.08);
  cursor: pointer;
  font-size: 13px;
  font-weight: 600;
  user-select: none;
}

.thinking-chain-header:hover {
  background: rgba(124, 106, 247, 0.15);
}

.llm-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: linear-gradient(135deg, #7c6af7, #a78bfa);
  flex-shrink: 0;
}

.tc-toggle {
  margin-left: auto;
  font-size: 12px;
  color: #7c6af7;
}

.thinking-chain-content pre {
  margin: 0;
  padding: 14px 16px;
  font-size: 12px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
  background: rgba(124, 106, 247, 0.04);
  border-top: 1px solid rgba(124, 106, 247, 0.15);
}
</style>
