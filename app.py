import sys
import subprocess
import threading

# --- STEP 1: Dependencies Check & Automated Installation ---
libs = {
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "numpy": "numpy",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
}
for module, package in libs.items():
    try:
        __import__(module)
    except ImportError:
        print(f"📦 Installing production dependency: {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import uvicorn
from datetime import datetime, timedelta
import random
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# Fix seeds for reliability
random.seed(42)
np.random.seed(42)

# Initialize FastAPI App
app = FastAPI(
    title="Enterprise Real-Time Recommendation API",
    description="Production-grade ML engine featuring live contextual inference and automated MLOps validation gatekeeping.",
    version="2.0.0"
)

# --- STEP 2: The Core Retail Infrastructure Engine ---
class RetailStreamingEnvironment:
    def __init__(self):
        self.products = {
            101: {"name": "Flagship Smartphone", "category": "Phones", "price": 999},
            102: {"name": "Mid-Range Smartphone", "category": "Phones", "price": 499},
            103: {"name": "Wireless Earbuds", "category": "Accessories", "price": 149},
            104: {"name": "Fast Charger Block", "category": "Accessories", "price": 29},
            105: {"name": "USB-C Charging Cable", "category": "Accessories", "price": 19},
            106: {"name": "Noise Cancelling Headphones", "category": "Audio", "price": 299},
            107: {"name": "Bluetooth Speaker", "category": "Audio", "price": 89},
            108: {"name": "4K Gaming Monitor", "category": "Monitors", "price": 449},
            109: {"name": "Mechanical Keyboard", "category": "Peripherals", "price": 119},
            110: {"name": "Wireless Ergonomic Mouse", "category": "Peripherals", "price": 79},
        }
        self.historical_data = self._generate_historical_base(num_events=1200)
        self.live_stream_buffer = []

    def flush_buffer(self):
        events = pd.DataFrame(self.live_stream_buffer)
        self.live_stream_buffer.clear()
        return events

    def _generate_historical_base(self, num_events):
        data = []
        start_time = datetime.now() - timedelta(days=2)
        for i in range(num_events):
            user_id = random.randint(1, 50)
            product_id = random.choice(list(self.products.keys()))
            action = random.choice(["view", "add_to_cart", "purchase"])
            timestamp = start_time + timedelta(seconds=i * 50)
            data.append({"user_id": user_id, "product_id": product_id, "action": action, "timestamp": timestamp})
        return pd.DataFrame(data)

env = RetailStreamingEnvironment()

# --- STEP 3: Recommendation Model Layer with Validation Gates ---
class StreamingRecommendationEngine:
    def __init__(self, product_catalog):
        self.catalog = product_catalog
        self.action_weights = {"view": 1, "add_to_cart": 2, "purchase": 5}
        self.interaction_matrix = None
        self.item_similarity_df = None

    def train(self, data_df):
        if data_df.empty:
            return
        df_copy = data_df.copy()
        df_copy['score'] = df_copy['action'].map(self.action_weights)
        user_item_scores = df_copy.groupby(['user_id', 'product_id'])['score'].sum().unstack(fill_value=0)
        
        for prod_id in self.catalog.keys():
            if prod_id not in user_item_scores.columns:
                user_item_scores[prod_id] = 0
                
        user_item_scores = user_item_scores.reindex(sorted(user_item_scores.columns), axis=1)
        self.interaction_matrix = user_item_scores
        
        item_sim = cosine_similarity(user_item_scores.T)
        self.item_similarity_df = pd.DataFrame(item_sim, index=user_item_scores.columns, columns=user_item_scores.columns)

    def evaluate_model(self, test_df, k=3) -> float:
        if test_df.empty or self.item_similarity_df is None:
            return 0.0
        hits = 0
        total_evaluations = 0
        grouped_test = test_df.groupby('user_id')['product_id'].apply(list).to_dict()
        
        for user_id, true_items in grouped_test.items():
            if user_id not in self.interaction_matrix.index:
                continue
            recommendations = self.get_raw_recommendations(user_id, top_n=k)
            if any(item in recommendations for item in true_items):
                hits += 1
            total_evaluations += 1
            
        return float(hits / total_evaluations) if total_evaluations > 0 else 0.0

    def get_raw_recommendations(self, user_id, top_n=3) -> List[int]:
        if self.item_similarity_df is None or user_id not in self.interaction_matrix.index:
            return random.sample(list(self.catalog.keys()), top_n)
            
        user_vector = self.interaction_matrix.loc[user_id].values.reshape(1, -1)
        base_scores = np.dot(user_vector, self.item_similarity_df.values).flatten()
        recommendation_scores = pd.Series(base_scores, index=self.item_similarity_df.columns)
        
        already_interacted = self.interaction_matrix.loc[user_id]
        already_bought = already_interacted[already_interacted >= 5].index
        recommendation_scores = recommendation_scores.drop(already_bought, errors='ignore')
        
        return recommendation_scores.nlargest(top_n).index.tolist()

    def recommend(self, user_id, live_context_events: Optional[List[Dict]] = None, top_n=3) -> List[Dict]:
        if self.item_similarity_df is None or user_id not in self.interaction_matrix.index:
            random_ids = random.sample(list(self.catalog.keys()), top_n)
            return [self.catalog[pid] | {"product_id": pid} for pid in random_ids]

        user_vector = self.interaction_matrix.loc[user_id].values.reshape(1, -1)
        base_scores = np.dot(user_vector, self.item_similarity_df.values).flatten()
        recommendation_scores = pd.Series(base_scores, index=self.item_similarity_df.columns)
        
        if live_context_events:
            for event in live_context_events:
                if event['user_id'] == user_id:
                    pid = event['product_id']
                    if pid in self.item_similarity_df.index:
                        boost = self.action_weights.get(event['action'], 1) * 1.75
                        recommendation_scores += self.item_similarity_df[pid] * boost

        already_interacted = self.interaction_matrix.loc[user_id]
        already_bought = already_interacted[already_interacted >= 5].index
        recommendation_scores = recommendation_scores.drop(already_bought, errors='ignore')
        
        top_product_ids = recommendation_scores.nlargest(top_n).index.tolist()
        return [self.catalog[pid] | {"product_id": pid} for pid in top_product_ids if pid in self.catalog]

active_model = StreamingRecommendationEngine(env.products)
active_model.train(env.historical_data)

# --- STEP 4: MLOps Automated Governance Pipeline ---
class AutomatedMLOpsPipeline:
    def __init__(self, environment, model_layer):
        self.env = environment
        self.active_model = model_layer
        self.master_dataset = environment.historical_data.copy()
        self.model_version = 1
        self.performance_threshold = 0.15

    def run_training_and_gating_cycle(self) -> Dict:
        new_streaming_events = self.env.flush_buffer()
        if new_streaming_events.empty:
            return {"status": "skipped", "reason": "No data found inside streaming buffers."}
            
        print("⚡ MLOps Ingestion Gate Activated...")
        mask = np.random.rand(len(new_streaming_events)) < 0.7
        train_delta = new_streaming_events[mask]
        eval_delta = new_streaming_events[~mask]
        
        if train_delta.empty or eval_delta.empty:
            train_delta = new_streaming_events
            eval_delta = self.master_dataset.sample(n=min(50, len(self.master_dataset)))

        candidate_model = StreamingRecommendationEngine(self.env.products)
        potential_dataset = pd.concat([self.master_dataset, train_delta], ignore_index=True)
        candidate_model.train(potential_dataset)
        
        hit_rate = candidate_model.evaluate_model(eval_delta, k=3)
        print(f"🔬 Candidate Hit Rate@3: {hit_rate:.4f} vs Threshold: {self.performance_threshold}")
        
        if hit_rate >= self.performance_threshold:
            self.master_dataset = potential_dataset
            self.active_model.interaction_matrix = candidate_model.interaction_matrix
            self.active_model.item_similarity_df = candidate_model.item_similarity_df
            self.model_version += 1
            print(f"🚀 Model Hot-Swapped to V{self.model_version}.")
            return {"status": "success", "deployed_version": f"V{self.model_version}", "metric_hit_rate": f"{hit_rate:.4f}"}
        else:
            print("🚨 DEPLOYMENT GATE REJECTED. Rollback executed.")
            return {"status": "failed_validation_rollback", "current_version": f"V{mlops.model_version}", "metric_hit_rate": f"{hit_rate:.4f}"}

mlops = AutomatedMLOpsPipeline(env, active_model)

# --- STEP 5: FastAPI Request/Response Schema Definitions (Fixed Deprecations) ---
class UserEvent(BaseModel):
    user_id: int = Field(..., description="Target customer identifier unique tracking key", json_schema_extra={"example": 12})
    product_id: int = Field(..., description="Catalog item code ID identifier", json_schema_extra={"example": 105})
    action: str = Field(..., description="Interaction taxonomy tier. Must match: view, add_to_cart, purchase", json_schema_extra={"example": "view"})

class RecommendationResponse(BaseModel):
    product_id: int
    name: str
    category: str
    price: float

# --- STEP 6: API Endpoint Definition Routes ---
@app.get("/", tags=["Health"])
def API_root_status():
    return {
        "status": "online",
        "active_model_version": f"V{mlops.model_version}",
        "buffered_events_awaiting_mlops": len(env.live_stream_buffer)
    }

@app.post("/event", status_code=202, tags=["Data Ingestion"])
def track_user_interaction_event(event: UserEvent):
    if event.action not in ["view", "add_to_cart", "purchase"]:
        raise HTTPException(status_code=400, detail="Invalid action value type submitted.")
    if event.product_id not in env.products:
        raise HTTPException(status_code=404, detail="Product ID submitted does not exist.")
        
    payload = {
        "user_id": event.user_id,
        "product_id": event.product_id,
        "action": event.action,
        "timestamp": datetime.now()
    }
    env.live_stream_buffer.append(payload)
    return {"status": "accepted_queued", "buffer_current_size": len(env.live_stream_buffer)}

@app.get("/recommend", response_model=List[RecommendationResponse], tags=["Real-Time Inference Serving"])
def get_realtime_recommendations(user_id: int, limit: int = 3):
    return active_model.recommend(user_id=user_id, live_context_events=env.live_stream_buffer, top_n=limit)

@app.post("/mlops/train", tags=["Automated Governance Layer"])
def force_trigger_mlops_retrain_cycle(background_tasks: BackgroundTasks):
    if not env.live_stream_buffer:
        return {"status": "ignored", "reason": "Stream ingestion buffer is empty."}
    background_tasks.add_task(mlops.run_training_and_gating_cycle)
    return {"status": "pipeline_execution_initialized"}

# --- STEP 7: Local Server Launch Thread Execution ---
def run_uvicorn():
    config = uvicorn.Config(app=app, host="127.0.0.1", port=8000, log_level="info")
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌍 STARTING LOCAL PRODUCTION APP SERVER IN BACKGROUND THREAD")
    print("🔗 Local Server Port Inbound Address active on: http://127.0.0.1:8000")
    print("="*70 + "\n")

    # Run inside a detached thread to prevent halting the Colab ecosystem notebook loop
    server_thread = threading.Thread(target=run_uvicorn, daemon=True)
    server_thread.start()