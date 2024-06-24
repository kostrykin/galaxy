<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';

const jobs = ref([]);

const jobCount = computed(() => jobs.value.length);

onMounted(() => {
	console.log('UserJobsList component is mounted');
});
</script>

<template>
	<div>
    	<h1>User Jobs List</h1>
    	<p>{{ jobCount }} jobs loaded.</p>
	</div>
</template>
