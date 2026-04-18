# MicroLab-Nexus

A dynamic and interactive FastAPI-based web application for real-time laboratory collaboration, messaging, file sharing, polls, and dashboard features.

## Features

- Real-time WebSocket chat
- Private messaging
- File upload and sharing
- Poll creation and voting
- Leaderboard and activity feed
- Persistent login and session handling
- FastAPI + Uvicorn backend
- Render-ready deployment

## Deployment

Run with:

uvicorn microlab_nexus:app --host 0.0.0.0 --port $PORT