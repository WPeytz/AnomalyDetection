"""
Generate architecture diagram for DINOv3 Prompt Adaptation using Graphviz.
"""

from graphviz import Digraph


def create_prompt_architecture_diagram():
    """Create the main architecture diagram."""

    dot = Digraph(comment='DINOv3 Prompt Adaptation Architecture')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='0.6')
    dot.attr('node', fontname='Helvetica', fontsize='11')
    dot.attr('edge', fontname='Helvetica', fontsize='10')

    # Define node styles
    input_style = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#E3F2FD', 'color': '#1976D2'}
    frozen_style = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FFF9C4', 'color': '#F9A825'}
    learnable_style = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FFECB3', 'color': '#FF8F00', 'penwidth': '2'}
    operation_style = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#E8EAF6', 'color': '#3F51B5'}
    embedding_style = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#C8E6C9', 'color': '#388E3C'}
    loss_style = {'shape': 'diamond', 'style': 'filled', 'fillcolor': '#FFCDD2', 'color': '#D32F2F'}
    inference_style = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#F5F5F5', 'color': '#616161'}
    output_style = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FFCCBC', 'color': '#E64A19'}

    # Main architecture subgraph
    with dot.subgraph(name='cluster_main') as main:
        main.attr(label='DINOv3 Prompt Adaptation', labelloc='t', fontsize='14', fontname='Helvetica-Bold',
                  style='rounded', color='#424242', bgcolor='#FAFAFA')

        # Input
        main.node('input', 'Input Image\n(224 × 224 × 3)', **input_style)

        # Frozen DINOv3
        main.node('dinov3', 'Frozen DINOv3\n(ViT-S/16)', **frozen_style)

        # Patch embeddings
        main.node('patches', 'Patch Embeddings\n[B, 196, 384]', **embedding_style)

        # Learnable prompts
        main.node('prompts', 'Learnable Prompts\n[10, 384]\n(× prompt_scales)', **learnable_style)

        # Attention computation
        main.node('normalize', 'L2 Normalize\n(patches & prompts)', **operation_style)

        main.node('attention', 'Attention Weights\nsoftmax(patches @ prompts.T)\n[B, 196, 10]', **operation_style)

        main.node('weighted_sum', 'Weighted Sum\nattn @ prompts\n[B, 196, 384]', **operation_style)

        # Residual connection
        main.node('scale', 'Scale (α=0.1)', **operation_style)

        main.node('add', '+', shape='circle', style='filled', fillcolor='white', color='#424242', width='0.4', fixedsize='true')

        # Output embeddings
        main.node('adapted', 'Adapted Embeddings\n[B, 196, 384]', **embedding_style)

        # Edges in main flow
        main.edge('input', 'dinov3')
        main.edge('dinov3', 'patches')
        main.edge('patches', 'normalize')
        main.edge('prompts', 'normalize')
        main.edge('normalize', 'attention')
        main.edge('attention', 'weighted_sum')
        main.edge('prompts', 'weighted_sum', style='dashed', color='#FF8F00')
        main.edge('weighted_sum', 'scale')
        main.edge('scale', 'add')
        main.edge('patches', 'add', label='residual', style='dashed', color='#388E3C')
        main.edge('add', 'adapted')

    # Training subgraph
    with dot.subgraph(name='cluster_training') as train:
        train.attr(label='Training (Few-Shot)', labelloc='t', fontsize='12', fontname='Helvetica-Bold',
                   style='rounded,dashed', color='#D32F2F', bgcolor='#FFF8E1')

        train.node('normal_emb', 'Normal\nEmbeddings', **embedding_style)
        train.node('defect_emb', 'Defect\nEmbeddings', **embedding_style)
        train.node('sep_loss', 'Separability\nLoss', **loss_style)

        train.edge('normal_emb', 'sep_loss')
        train.edge('defect_emb', 'sep_loss')

    # Inference subgraph
    with dot.subgraph(name='cluster_inference') as infer:
        infer.attr(label='Inference & Segmentation', labelloc='t', fontsize='12', fontname='Helvetica-Bold',
                   style='rounded', color='#616161', bgcolor='#ECEFF1')

        infer.node('memory', 'Memory Bank\n(Normal Embeddings)', **inference_style)
        infer.node('knn', 'k-NN Distance\nScoring', shape='diamond', style='filled', fillcolor='#E0E0E0', color='#616161')
        infer.node('heatmap', 'Anomaly Map\n(Heatmap)', **output_style)
        infer.node('sam', 'SAM\n(Segmentation)', **output_style)

        infer.edge('memory', 'knn', label='reference')
        infer.edge('knn', 'heatmap')
        infer.edge('heatmap', 'sam')

    # Connect subgraphs
    dot.edge('adapted', 'normal_emb', style='dashed', color='#388E3C')
    dot.edge('adapted', 'defect_emb', style='dashed', color='#388E3C')
    dot.edge('sep_loss', 'prompts', label='update\n(backprop)', style='dashed', color='#D32F2F', constraint='false')

    dot.edge('adapted', 'knn', label='test embeds', color='#424242')

    return dot


def create_per_patch_detail_diagram():
    """Create detailed diagram of per-patch modulation."""

    dot = Digraph(comment='Per-Patch Modulation Detail')
    dot.attr(rankdir='LR', splines='ortho', nodesep='0.4', ranksep='0.5')
    dot.attr('node', fontname='Helvetica', fontsize='10')

    op_style = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#E8EAF6', 'color': '#3F51B5'}
    tensor_style = {'shape': 'box', 'style': 'filled', 'fillcolor': '#E0E0E0', 'color': '#757575'}

    with dot.subgraph(name='cluster_detail') as d:
        d.attr(label='Per-Patch Modulation (Default Mode)', fontsize='12', style='rounded', bgcolor='#FAFAFA')

        # Input tensors
        d.node('p_in', 'patches\n[B, N, D]', **tensor_style)
        d.node('pr_in', 'prompts\n[B, P, D]', **tensor_style)

        # Operations
        d.node('norm1', 'L2 Norm', **op_style)
        d.node('norm2', 'L2 Norm', **op_style)
        d.node('matmul1', 'patches @ prompts.T\n[B, N, P]', **op_style)
        d.node('softmax', 'Softmax\n(dim=-1)', **op_style)
        d.node('matmul2', 'attn @ prompts\n[B, N, D]', **op_style)
        d.node('scale', '× α', **op_style)
        d.node('add', '+', shape='circle', style='filled', fillcolor='white', width='0.3')
        d.node('out', 'output\n[B, N, D]', **tensor_style)

        # Flow
        d.edge('p_in', 'norm1')
        d.edge('pr_in', 'norm2')
        d.edge('norm1', 'matmul1')
        d.edge('norm2', 'matmul1')
        d.edge('matmul1', 'softmax')
        d.edge('softmax', 'matmul2')
        d.edge('pr_in', 'matmul2', style='dashed')
        d.edge('matmul2', 'scale')
        d.edge('scale', 'add')
        d.edge('p_in', 'add', style='dashed', label='residual')
        d.edge('add', 'out')

    return dot


def create_learned_transform_diagram():
    """Create diagram for learned_transform mode (optional)."""

    dot = Digraph(comment='Learned Transform Mode')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.4', ranksep='0.4')
    dot.attr('node', fontname='Helvetica', fontsize='10')

    op_style = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#E8EAF6', 'color': '#3F51B5'}
    mlp_style = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#FFECB3', 'color': '#FF8F00', 'penwidth': '2'}
    tensor_style = {'shape': 'box', 'style': 'filled', 'fillcolor': '#E0E0E0', 'color': '#757575'}

    with dot.subgraph(name='cluster_transform') as d:
        d.attr(label='Learned Transform Mode (Optional)', fontsize='12', style='rounded,dashed', bgcolor='#FFF8E1')

        d.node('patches', 'patches [B, N, D]', **tensor_style)
        d.node('prompt_effect', 'prompt_effect [B, N, D]', **tensor_style)
        d.node('concat', 'Concat\n[B, N, 2D]', **op_style)
        d.node('mlp', 'Transform MLP\nLinear(2D→D) → GELU → Linear(D→D)', **mlp_style)
        d.node('scale', '× α', **op_style)
        d.node('add', '+', shape='circle', style='filled', fillcolor='white', width='0.3')
        d.node('out', 'output [B, N, D]', **tensor_style)

        d.edge('patches', 'concat')
        d.edge('prompt_effect', 'concat')
        d.edge('concat', 'mlp')
        d.edge('mlp', 'scale')
        d.edge('scale', 'add')
        d.edge('patches', 'add', style='dashed', label='residual')
        d.edge('add', 'out')

    return dot


if __name__ == '__main__':
    import os

    output_dir = os.path.dirname(os.path.abspath(__file__))

    # Generate main architecture
    print("Generating main architecture diagram...")
    main_diagram = create_prompt_architecture_diagram()
    main_path = os.path.join(output_dir, 'architecture_main')

    # Try to render, fall back to saving DOT source
    try:
        main_diagram.render(main_path, format='png', cleanup=True)
        main_diagram.render(main_path, format='pdf', cleanup=True)
        print(f"  Saved: {main_path}.png and {main_path}.pdf")
    except Exception as e:
        # Save DOT source file instead
        dot_path = main_path + '.dot'
        main_diagram.save(dot_path)
        print(f"  Graphviz not installed. Saved DOT source: {dot_path}")
        print(f"  Render online at: https://dreampuf.github.io/GraphvizOnline/")

    # Generate per-patch detail
    print("Generating per-patch detail diagram...")
    detail_diagram = create_per_patch_detail_diagram()
    detail_path = os.path.join(output_dir, 'architecture_per_patch')
    try:
        detail_diagram.render(detail_path, format='png', cleanup=True)
        print(f"  Saved: {detail_path}.png")
    except Exception:
        dot_path = detail_path + '.dot'
        detail_diagram.save(dot_path)
        print(f"  Saved DOT source: {dot_path}")

    # Generate learned transform diagram
    print("Generating learned transform diagram...")
    transform_diagram = create_learned_transform_diagram()
    transform_path = os.path.join(output_dir, 'architecture_learned_transform')
    try:
        transform_diagram.render(transform_path, format='png', cleanup=True)
        print(f"  Saved: {transform_path}.png")
    except Exception:
        dot_path = transform_path + '.dot'
        transform_diagram.save(dot_path)
        print(f"  Saved DOT source: {dot_path}")

    print("\nDone! All diagrams generated.")
    print("\nTo render DOT files to PNG:")
    print("  1. Install Graphviz: brew install graphviz")
    print("  2. Run: dot -Tpng architecture_main.dot -o architecture_main.png")
    print("  Or use online: https://dreampuf.github.io/GraphvizOnline/")
