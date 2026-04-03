SESSION_NAME="qwen_download"

tmux kill-session -t $SESSION_NAME 2>/dev/null

tmux new-session -d -s $SESSION_NAME

tmux send-keys -t $SESSION_NAME 'while true; do modelscope download --model Qwen/Qwen-Image --local_dir data/qwen_image && break; echo "Download failed, retrying in 5s..."; sleep 5;  done' C-m

echo "Download started in tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"ø
