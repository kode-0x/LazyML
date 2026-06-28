"""Ensemble learning builders for voting and stacking."""

from typing import Any

import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge


def build_voting_ensemble(
    models: list[tuple[str, Any]],
    task_type: str,
    voting: str = "soft",
) -> VotingClassifier | None:
    """
    Build a voting ensemble from top models.
    
    Args:
        models: List of (name, model) tuples for ensemble
        task_type: "classification" or "regression"
        voting: "hard" or "soft" for classification
        
    Returns:
        VotingClassifier for classification, None for regression
    """
    if task_type != "classification":
        return None
    
    if len(models) < 2:
        return None
    
    # Filter to models that support predict_proba for soft voting
    if voting == "soft":
        valid_models = []
        for name, model in models:
            if hasattr(model, "predict_proba"):
                valid_models.append((name, model))
        
        if len(valid_models) < 2:
            # Fall back to hard voting if not enough models support probabilities
            voting = "hard"
            models = models[:5]  # Use top 5 for hard voting
        else:
            models = valid_models[:5]
    
    ensemble = VotingClassifier(
        estimators=models,
        voting=voting,
        n_jobs=-1,
    )
    
    return ensemble


def build_stacking_ensemble(
    models: list[tuple[str, Any]],
    task_type: str,
    cv: int = 5,
    passthrough: bool = False,
) -> StackingClassifier | StackingRegressor | None:
    """
    Build a stacking ensemble from top models.
    
    Args:
        models: List of (name, model) tuples as base learners
        task_type: "classification" or "regression"
        cv: Number of folds for stacking
        passthrough: Whether to pass original features to meta-learner
        
    Returns:
        StackingClassifier or StackingRegressor
    """
    if len(models) < 2:
        return None
    
    # Use top 5 models as base learners
    base_learners = models[:5]
    
    # Select meta-learner
    if task_type == "classification":
        # Use LogisticRegression as meta-learner for classification
        meta_learner = LogisticRegression(max_iter=1000, random_state=42)
        ensemble = StackingClassifier(
            estimators=base_learners,
            final_estimator=meta_learner,
            cv=cv,
            passthrough=passthrough,
            n_jobs=-1,
        )
    else:
        # Use Ridge regression as meta-learner for regression
        meta_learner = Ridge(random_state=42)
        ensemble = StackingRegressor(
            estimators=base_learners,
            final_estimator=meta_learner,
            cv=cv,
            passthrough=passthrough,
            n_jobs=-1,
        )
    
    return ensemble


def create_ensemble_from_results(
    results: dict[str, dict[str, Any]],
    task_type: str,
    ensemble_type: str = "voting",
    top_n: int = 3,
) -> tuple[Any, list[str]] | None:
    """
    Create ensemble from training results.
    
    Args:
        results: Training results dict from ModelTrainer
        task_type: "classification" or "regression"
        ensemble_type: "voting" or "stacking"
        top_n: Number of top models to include
        
    Returns:
        Tuple of (ensemble_model, list_of_model_names) or None
    """
    # Filter to successfully fitted models
    fitted_models = [
        (name, result["model"]) 
        for name, result in results.items() 
        if result.get("fitted", False)
    ]
    
    if len(fitted_models) < 2:
        return None
    
    # Sort by cv_score (higher is better for both classification and regression in our setup)
    fitted_models.sort(
        key=lambda x: results[x[0]]["cv_score"],
        reverse=True
    )
    
    # Take top N
    top_models = fitted_models[:top_n]
    model_names = [name for name, _ in top_models]
    
    if ensemble_type == "voting" and task_type == "classification":
        ensemble = build_voting_ensemble(top_models, task_type, voting="soft")
    elif ensemble_type == "stacking":
        ensemble = build_stacking_ensemble(top_models, task_type, cv=5)
    else:
        return None
    
    if ensemble is None:
        return None
    
    return ensemble, model_names
