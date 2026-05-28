<template>
  <section class="panel">
    <div class="panel-head">
      <div>
        <p class="eyebrow">历史任务</p>
        <h2>最近检测记录</h2>
      </div>
      <button class="ghost-button" @click="$emit('refresh')">刷新</button>
    </div>
    <div v-if="!tasks.length" class="empty-state">还没有检测记录。</div>
    <button
      v-for="task in tasks"
      :key="task.task_id"
      class="history-item"
      :class="{ active: task.task_id === activeTaskId }"
      @click="$emit('select', task.task_id)"
    >
      <div class="history-top">
        <strong>{{ task.filename }}</strong>
        <StatusBadge :text="task.label || task.status" />
      </div>
      <div class="history-meta">
        <span>{{ task.stage }}</span>
        <span v-if="task.confidence !== null && task.confidence !== undefined">
          置信度 {{ Number(task.confidence).toFixed(2) }}
        </span>
      </div>
    </button>
  </section>
</template>

<script setup>
import StatusBadge from "./StatusBadge.vue";

defineProps({
  tasks: {
    type: Array,
    default: () => []
  },
  activeTaskId: {
    type: String,
    default: ""
  }
});

defineEmits(["select", "refresh"]);
</script>

