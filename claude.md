# Plateforme IA Autonome — Phase 1

## Stack technique décidée
- Framework agentique : LangGraph 0.2 (D04 — DÉCIDÉ)
- Message broker : Redis Streams (D01 — DÉCIDÉ)
- LLM Gateway : LiteLLM Proxy
- Backend : FastAPI Python 3.11
- Frontend : Next.js 14 + Tailwind CSS
- GPU infra : OVH/Scaleway cloud souverain (D02 — DÉCIDÉ)

## Contraintes critiques
- Circuit breaker OBLIGATOIRE : max_iterations=3 dans chaque noeud LangGraph (R01)
- Tout code doit passer par le QA Worker avant le Sandbox (R06)
- Aucun port GPU exposé publiquement — tout via VPN Wireguard (Gate G7)

## Conventions de code
- Tests unitaires obligatoires pour chaque endpoint FastAPI
- Logging structuré JSON sur tous les workers
- Variables d'environnement dans .env.example (jamais de secrets hardcodés)

## Gate de sortie Phase 1
- G1 : voix → spec < 60s
- G4 : 0 boucle infinie sur 20 tests
- G6 : LLaMA 70B P95 < 2s
