<template>
  <section class="panel">
    <div class="panel-head">
      <div>
        <p class="eyebrow">上传检测</p>
        <h2>提交一张待检测的人脸图像</h2>
      </div>
      <StatusBadge :text="uploading ? 'running' : 'ready'" />
    </div>
    <label class="upload-zone">
      <input type="file" accept=".jpg,.jpeg,.png,.webp" @change="onFileChange" />
      <span class="upload-title">{{ selectedFile ? selectedFile.name : "点击选择图片" }}</span>
      <span class="upload-tip">支持 JPG / PNG / WEBP，单张不超过 10MB。</span>
    </label>

    <div class="sample-gallery">
      <p class="eyebrow sample-title">示例图片（点击可快速选用）</p>
      <div class="sample-grid">
        <button
          v-for="s in samples"
          :key="s.file"
          class="sample-thumb"
          :class="{ active: selectedFile && selectedFile.name === s.file }"
          :title="s.label"
          @click="pickSample(s)"
        >
          <img :src="'/samples/' + s.file" :alt="s.label" />
          <span class="sample-label">{{ s.label }}</span>
        </button>
      </div>
    </div>

    <button class="primary-button" :disabled="!selectedFile || uploading" @click="submit">
      {{ uploading ? "任务创建中..." : "开始检测" }}
    </button>
    <p v-if="error" class="error-text">{{ error }}</p>
  </section>
</template>

<script setup>
import { ref } from "vue";
import StatusBadge from "./StatusBadge.vue";

defineProps({
  uploading: {
    type: Boolean,
    default: false
  }
});

const emit = defineEmits(["submit"]);
const selectedFile = ref(null);
const error = ref("");

const samples = [
  { file: "real_portrait_1.jpg", label: "真实肖像 1" },
  { file: "real_portrait_2.jpg", label: "真实肖像 2" },
  { file: "real_portrait_3.jpg", label: "真实肖像 3" },
  { file: "real_portrait_4.jpg", label: "真实肖像 4" },
  { file: "real_group.jpg", label: "多人场景" },
  { file: "ai_generated_1.jpg", label: "AI 生成" }
];

async function pickSample(sample) {
  error.value = "";
  try {
    const resp = await fetch("/samples/" + sample.file);
    const blob = await resp.blob();
    selectedFile.value = new File([blob], sample.file, { type: blob.type });
  } catch {
    error.value = "加载示例图片失败。";
  }
}

function onFileChange(event) {
  error.value = "";
  const [file] = event.target.files || [];
  selectedFile.value = file || null;
}

function submit() {
  if (!selectedFile.value) {
    error.value = "请先选择一张图片。";
    return;
  }
  error.value = "";
  emit("submit", selectedFile.value);
}
</script>
