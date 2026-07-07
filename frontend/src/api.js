import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL
  ? `${import.meta.env.VITE_API_URL}/api`
  : '/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 60000,
});

export const predict = async (data) => {
  const response = await api.post('/predict', data);
  return response.data;
};

export const getProducts = async () => {
  const response = await api.get('/products');
  return response.data;
};

export const getAllProducts = async () => {
  const response = await api.get('/products/all');
  return response.data;
};

export const getStores = async () => {
  const response = await api.get('/stores');
  return response.data;
};

export const getHistory = async (limit = 20) => {
  const response = await api.get('/history', { params: { limit } });
  return response.data;
};

export const getAnalytics = async () => {
  const response = await api.get('/analytics');
  return response.data;
};

export const getHeatmap = async () => {
  const response = await api.get('/heatmap');
  return response.data;
};

export const generateReport = async (data) => {
  const response = await api.post('/report', data);
  return response.data;
};

export default api;
