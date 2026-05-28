<template>
  <span class="status-badge" :data-status="tone">{{ text }}</span>
</template>

<script setup>
import { computed } from "vue";

const props = defineProps({
  text: {
    type: String,
    required: true
  }
});

const tone = computed(() => {
  if (props.text.includes("伪造") || props.text === "failed") return "danger";
  if (props.text.includes("复核") || props.text === "running") return "warning";
  if (props.text.includes("真实") || props.text === "completed") return "success";
  return "neutral";
});
</script>

