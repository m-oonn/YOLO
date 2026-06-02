<template>
  <div class="filter-bar">
    <div class="filter-left">
      <template v-for="filter in filters" :key="filter.key">
        <el-select
          v-if="filter.type === 'select'"
          v-model="filterValues[filter.key]"
          :placeholder="filter.label"
          size="default"
          clearable
          class="filter-select"
          @change="onSearch"
        >
          <el-option
            v-for="opt in filter.options"
            :key="opt.value"
            :label="opt.label"
            :value="opt.value"
          />
        </el-select>
        <el-input
          v-else-if="filter.type === 'input'"
          v-model="filterValues[filter.key]"
          :placeholder="filter.label"
          size="default"
          clearable
          class="filter-input"
          @keyup.enter="onSearch"
        />
      </template>
      <el-button type="primary" size="default" @click="onSearch">
        <el-icon><Search /></el-icon>
        搜索
      </el-button>
      <el-button size="default" @click="onReset">
        <el-icon><RefreshRight /></el-icon>
        重置
      </el-button>
    </div>
    <div v-if="$slots.right" class="filter-right">
      <slot name="right" />
    </div>
  </div>
</template>

<script setup>
import { reactive } from 'vue'
import { Search, RefreshRight } from '@element-plus/icons-vue'

const props = defineProps({
  filters: {
    type: Array,
    default: () => [],
  },
})

const emit = defineEmits(['search', 'reset'])

const filterValues = reactive({})

const initValues = () => {
  props.filters.forEach((f) => {
    filterValues[f.key] = ''
  })
}

initValues()

const onFilterChange = () => {
  // auto-search on change for selects, debounced for inputs
}

const onSearch = () => {
  const params = {}
  Object.entries(filterValues).forEach(([key, val]) => {
    if (val !== '' && val !== null && val !== undefined) {
      params[key] = val
    }
  })
  emit('search', params)
}

const onReset = () => {
  props.filters.forEach((f) => {
    filterValues[f.key] = ''
  })
  emit('reset')
}
</script>

<style scoped>
.filter-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}

.filter-left {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  flex: 1;
}

.filter-right {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
}

.filter-select {
  width: 150px;
}

.filter-input {
  width: 200px;
}

@media (max-width: 768px) {
  .filter-left {
    width: 100%;
  }
  .filter-select,
  .filter-input {
    flex: 1;
    min-width: 120px;
  }
  .filter-bar {
    flex-direction: column;
    align-items: stretch;
  }
  .filter-right {
    width: 100%;
    justify-content: flex-end;
  }
}
</style>
