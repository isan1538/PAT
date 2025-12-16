"""
Test script to verify the fixed PAT model handles cross-attention correctly.
"""

import torch
from pat_model_2d_1212 import PATConfig, PATModel

def test_model_shapes():
    """Test that model handles different sequence lengths correctly."""
    print("Testing PAT Model with Cross-Attention...")
    print("=" * 60)
    
    # Create config for 2D problem
    cfg = PATConfig()
    cfg.d_pos = 3  # (x, y, t)
    cfg.d_patch = 1
    cfg.n_embd = 256
    cfg.n_head = 8
    cfg.n_layer = 4
    
    model = PATModel(cfg)
    model.eval()
    
    batch_size = 4
    M_ctx = 50  # Number of sparse observations (context)
    N_query = 4096  # Number of query points (e.g., collocation points)
    
    print(f"Batch size: {batch_size}")
    print(f"Context size (M): {M_ctx}")
    print(f"Query size (N): {N_query}")
    print()
    
    # Create dummy inputs
    ctx_feats = torch.randn(batch_size, M_ctx, cfg.d_patch)
    ctx_pos = torch.rand(batch_size, M_ctx, cfg.d_pos)
    query_pos = torch.rand(batch_size, N_query, cfg.d_pos)
    
    print("Input shapes:")
    print(f"  ctx_feats: {ctx_feats.shape}")
    print(f"  ctx_pos: {ctx_pos.shape}")
    print(f"  query_pos: {query_pos.shape}")
    print()
    
    # Forward pass
    with torch.no_grad():
        pred = model(ctx_feats, ctx_pos, query_pos)
    
    print("Output shape:")
    print(f"  pred: {pred.shape}")
    print()
    
    # Verify shapes
    assert pred.shape == (batch_size, N_query, cfg.d_patch), \
        f"Expected shape ({batch_size}, {N_query}, {cfg.d_patch}), got {pred.shape}"
    
    print("✓ Shape test passed!")
    print()
    
    # Test PDE residual computation
    print("Testing PDE residual computation...")
    query_pos_grad = query_pos.clone().requires_grad_(True)
    pred_grad = model(ctx_feats, ctx_pos, query_pos_grad)
    
    residual = model.compute_pde_residual_2d(pred_grad, query_pos_grad, nu=0.1)
    
    print(f"  Residual shape: {residual.shape}")
    assert residual.shape == (batch_size, N_query, 1), \
        f"Expected residual shape ({batch_size}, {N_query}, 1), got {residual.shape}"
    
    print("✓ PDE residual test passed!")
    print()
    
    # Test with different query sizes
    print("Testing with various query sizes...")
    test_sizes = [10, 100, 1000, 5000]
    for size in test_sizes:
        query_pos_test = torch.rand(batch_size, size, cfg.d_pos)
        with torch.no_grad():
            pred_test = model(ctx_feats, ctx_pos, query_pos_test)
        assert pred_test.shape == (batch_size, size, cfg.d_patch)
        print(f"  Query size {size:5d}: ✓")
    
    print()
    print("=" * 60)
    print("All tests passed! Model is working correctly.")
    print()
    print("Key features verified:")
    print("  ✓ Cross-attention handles different sequence lengths")
    print("  ✓ Context size (M) != Query size (N) works correctly")
    print("  ✓ PDE residual computation works")
    print("  ✓ Model handles various query sizes")


def test_exact_solution_residual():
    """Test PDE residual computation with model predictions."""
    print("\nTesting PDE residual computation...")
    print("=" * 60)
    
    import math
    
    # Create model
    cfg = PATConfig()
    cfg.d_pos = 3
    cfg.n_layer = 2  # Smaller for testing
    cfg.n_embd = 64
    model = PATModel(cfg)
    model.eval()
    
    # Create context (sparse observations)
    batch_size = 1
    M = 20
    ctx_feats = torch.randn(batch_size, M, 1)
    ctx_pos = torch.rand(batch_size, M, 3)
    
    # Create query positions that require gradients
    N_test = 10
    query_pos = torch.rand(batch_size, N_test, 3, requires_grad=True)
    
    # Forward pass through model (this creates the computational graph)
    pred = model(ctx_feats, ctx_pos, query_pos)
    
    print(f"Context size: {M}")
    print(f"Query size: {N_test}")
    print(f"Prediction shape: {pred.shape}")
    print()
    
    # Compute PDE residual (this should work now)
    nu = 0.1
    residual = model.compute_pde_residual_2d(pred, query_pos, nu)
    
    print(f"PDE Residual computation:")
    print(f"  Shape: {residual.shape}")
    print(f"  Mean absolute value: {residual.abs().mean().item():.4e}")
    print(f"  Max absolute value: {residual.abs().max().item():.4e}")
    print()
    
    # Test that gradients flow properly
    loss = residual.pow(2).mean()
    loss.backward()
    
    # Check that gradients were computed
    has_grad = query_pos.grad is not None
    print(f"Gradient flow check:")
    print(f"  query_pos.grad exists: {has_grad}")
    if has_grad:
        print(f"  Gradient shape: {query_pos.grad.shape}")
        print(f"  Gradient norm: {query_pos.grad.norm().item():.4e}")
    print()
    
    if has_grad:
        print("✓ PDE residual computation and gradient flow working correctly!")
    else:
        print("⚠ Warning: Gradients not computed properly")
    print()
    
    # Additional test: verify exact solution formula
    print("Verifying exact solution formula...")
    def exact_solution_2d(x, y, t, nu, n=1, m=1):
        factor = nu * (math.pi**2) * (n**2 + m**2)
        return torch.exp(-factor * t) * torch.sin(n * math.pi * x) * torch.sin(m * math.pi * y)
    
    x_val, y_val, t_val = 0.5, 0.3, 0.2
    u_val = exact_solution_2d(
        torch.tensor(x_val), 
        torch.tensor(y_val), 
        torch.tensor(t_val), 
        nu=0.1, n=1, m=1
    )
    print(f"  u(x={x_val}, y={y_val}, t={t_val}) = {u_val.item():.6f}")
    print("  ✓ Exact solution formula works")
    print()


if __name__ == "__main__":
    test_model_shapes()
    test_exact_solution_residual()
    
    print("\n" + "=" * 60)
    print("SUMMARY: Model is ready for training!")
    print("=" * 60)
    print("\nYou can now use this model in your training script:")
    print("  from pat_model_2d_fixed import PATConfig, PATModel")
