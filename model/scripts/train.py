"""
主训练脚本：Chunk-level离线强化学习
"""
import argparse
from pathlib import Path

from ..training import ChunkLevelOfflineRL


def main():
    parser = argparse.ArgumentParser(description="Chunk-level离线强化学习训练")
    parser.add_argument("--dataset-dir", type=str, required=True,
                       help="LeRobot数据集目录路径")
    parser.add_argument("--action-horizon", type=int, default=10,
                       help="动作chunk长度 H (默认: 10)")
    parser.add_argument("--action-dim", type=int, default=7,
                       help="动作维度 (默认: 7)")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="折扣因子 (默认: 0.99)")
    parser.add_argument("--beta", type=float, default=1.0,
                       help="Advantage weighting温度参数 (默认: 1.0)")
    parser.add_argument("--lambda-v", type=float, default=1.0,
                       help="Value损失权重 (默认: 1.0)")
    parser.add_argument("--lambda-pi", type=float, default=1.0,
                       help="Policy损失权重 (默认: 1.0)")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="学习率 (默认: 3e-4)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch大小 (默认: 32)")
    parser.add_argument("--num-epochs", type=int, default=100,
                       help="训练epochs数 (默认: 100)")
    parser.add_argument("--log-interval", type=int, default=100,
                       help="日志打印间隔 (默认: 100)")
    parser.add_argument("--save-interval", type=int, default=10,
                       help="Checkpoint保存间隔(epochs) (默认: 10)")
    parser.add_argument("--device", type=str, default=None,
                       help="设备 ('cpu' 或 'cuda', 默认: 自动检测)")
    parser.add_argument("--resume", type=str, default=None,
                       help="恢复训练的checkpoint路径")
    parser.add_argument("--use-pi0-actions", action="store_true",
                       help="使用Pi0生成动作chunk（而不是使用演示数据中的动作）")
    parser.add_argument("--pi0-checkpoint", type=str, default=None,
                       help="Pi0 checkpoint路径（默认使用pi05_libero）")
    parser.add_argument("--pi0-config", type=str, default=None,
                       help="Pi0配置名称（默认: pi05_libero）")
    parser.add_argument("--init-policy-from-pi0", action="store_true",
                       help="使用Pi0初始化策略网络（需要手动实现权重映射）")
    parser.add_argument("--pi0-checkpoints-dir", type=str, default=None,
                       help="Pi0 checkpoints本地目录（默认：checkpoints/pi0）")
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = ChunkLevelOfflineRL(
        dataset_dir=args.dataset_dir,
        action_horizon=args.action_horizon,
        action_dim=args.action_dim,
        gamma=args.gamma,
        beta=args.beta,
        lambda_v=args.lambda_v,
        lambda_pi=args.lambda_pi,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        use_pi0_actions=args.use_pi0_actions,
        pi0_checkpoint=args.pi0_checkpoint,
        pi0_config=args.pi0_config,
        init_policy_from_pi0=args.init_policy_from_pi0,
        pi0_checkpoints_dir=args.pi0_checkpoints_dir,
    )
    
    # 恢复训练（如果指定）
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train(
        num_epochs=args.num_epochs,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
    )


if __name__ == "__main__":
    main()

