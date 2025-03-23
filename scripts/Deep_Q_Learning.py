# Import các thư viện cần thiết
import numpy as np  # Thư viện xử lý mảng số học
import torch  # Thư viện PyTorch cho deep learning
import torch.nn as nn  # Module neural network của PyTorch
import torch.optim as optim  # Module tối ưu hóa (optimizer) của PyTorch
import random  # Thư viện tạo số ngẫu nhiên
from gym import Env  # Lớp cơ sở để xây dựng môi trường RL từ OpenAI Gym
from gym.spaces import Box  # Không gian liên tục cho action và state
from collections import deque  # Hàng đợi hai đầu để làm replay buffer
import matplotlib.pyplot as plt  # Thư viện vẽ đồ thị
import cv2  # Thư viện xử lý ảnh (OpenCV)
import os  # Thư viện thao tác với hệ thống file

# --- BƯỚC 1: TIỀN XỬ LÝ HEATMAP ---
def preprocess_heatmap(heatmap_path):  # Hàm tiền xử lý ảnh heatmap
    heatmap_rgb = cv2.imread(heatmap_path)  # Đọc ảnh từ đường dẫn, định dạng BGR
    heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)  # Chuyển sang định dạng RGB
    heatmap_gray = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2GRAY)  # Chuyển sang ảnh grayscale
    heatmap_gray = heatmap_gray / 255.0  # Chuẩn hóa giá trị pixel về [0, 1]
    heatmap_vector = heatmap_gray.flatten()  # Chuyển ma trận 2D thành vector 1D
    return heatmap_vector, heatmap_rgb  # Trả về vector và ảnh RGB gốc

# --- BƯỚC 2: XÂY DỰNG REPLAY BUFFER ---
class ReplayBuffer:  # Lớp lưu trữ kinh nghiệm để huấn luyện agent
    def __init__(self, max_size=2000):  # Khởi tạo với kích thước tối đa
        self.memory = deque(maxlen=max_size)  # Sử dụng deque với giới hạn kích thước
    
    def add(self, state, action, reward, next_state, done):  # Thêm một kinh nghiệm vào buffer
        self.memory.append((state, action, reward, next_state, done))  # Lưu tuple kinh nghiệm
    
    def sample(self, batch_size):  # Lấy mẫu ngẫu nhiên một batch từ buffer
        if len(self.memory) < batch_size:  # Nếu số kinh nghiệm ít hơn batch_size
            return None  # Trả về None (chưa đủ dữ liệu)
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)  # Chọn ngẫu nhiên không lặp
        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in minibatch])  # Tách tuple thành các list
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))  # Chuyển thành numpy array và trả về
    
    def __len__(self):  # Phương thức trả về độ dài hiện tại của buffer
        return len(self.memory)  # Số lượng kinh nghiệm đang lưu

# --- BƯỚC 3: XÂY DỰNG MÔI TRƯỜNG SLICING ---
class SlicingEnvironment(Env):  # Lớp môi trường RL tùy chỉnh, kế thừa từ Env
    def __init__(self, heatmap_vector, heatmap_shape, heatmap_rgb, coverage_threshold=0.9, max_steps=50):  # Khởi tạo môi trường
        super().__init__()  # Gọi hàm khởi tạo của lớp cha (Env)
        self.heatmap_vector = heatmap_vector  # Vector heatmap (1D)
        self.heatmap_shape = heatmap_shape  # Kích thước heatmap (height, width)
        self.heatmap_rgb = heatmap_rgb  # Ảnh RGB gốc của heatmap
        self.height, self.width = heatmap_shape  # Gán chiều cao và chiều rộng
        self.slices = []  # Danh sách các slice (mỗi slice là tuple: x, y, w, h)
        self.state_max_slices = 10  # Số slice tối đa trong state
        self.coverage_threshold = coverage_threshold  # Ngưỡng bao phủ tối thiểu (mặc định 0.9)
        self.max_steps = max_steps  # Số bước tối đa trong một episode
        self.current_step = 0  # Đếm số bước hiện tại
        self.max_slices = 10  # Giới hạn tối đa số slice trong môi trường
        
        heatmap = self.heatmap_vector.reshape(self.heatmap_shape)  # Chuyển vector thành ma trận 2D
        self.total_important_heatmap = np.sum(heatmap >= 0.5)  # Tổng giá trị vùng quan trọng (>= 0.5)
        self.covered_heatmap = np.zeros(self.heatmap_shape, dtype=np.float32)  # Ma trận lưu vùng đã bao phủ
        
        state_size = len(heatmap_vector) + (self.state_max_slices * 4) + 1  # Kích thước state: heatmap + slices + num_slices
        self.observation_space = Box(low=0, high=1, shape=(state_size,), dtype=np.float32)  # Không gian state (liên tục [0, 1])
        self.action_space = Box(low=np.array([0, 0, 0, 0, -0.1, -0.1]),  # Không gian hành động: [create_flag, slice_idx, x_ratio, y_ratio, w_ratio, h_ratio]
                                high=np.array([1, 10, 1, 1, 0.1, 0.1]),
                                shape=(6,), dtype=np.float32)
    
    def reset(self):  # Đặt lại môi trường về trạng thái ban đầu
        self.slices = []  # Bắt đầu với danh sách slice rỗng
        self.current_step = 0  # Đặt lại số bước
        self.covered_heatmap = np.zeros(self.heatmap_shape, dtype=np.float32)  # Đặt lại vùng bao phủ
        return self._get_state()  # Trả về trạng thái ban đầu
    
    def _get_state(self):  # Tạo vector trạng thái cho agent
        heatmap_vec = self.heatmap_vector  # Lấy vector heatmap
        recent_slices = self.slices[-self.state_max_slices:] if len(self.slices) > self.state_max_slices else self.slices  # Lấy tối đa state_max_slices slice gần nhất
        all_slices = np.array(recent_slices) / np.array([self.width, self.height, self.width, self.height]) if recent_slices else np.zeros((self.state_max_slices, 4))  # Chuẩn hóa tọa độ slice
        all_slices_flat = all_slices.flatten()  # Chuyển thành vector 1D
        if len(all_slices_flat) < self.state_max_slices * 4:  # Nếu không đủ chiều dài
            all_slices_flat = np.pad(all_slices_flat, (0, self.state_max_slices * 4 - len(all_slices_flat)), 'constant')  # Đệm bằng 0
        num_slices = min(len(self.slices) / self.state_max_slices, 1.0)  # Tỷ lệ số slice hiện tại (0-1)
        return np.concatenate([heatmap_vec, all_slices_flat, [num_slices]]).astype(np.float32)  # Ghép heatmap, slices, num_slices thành state

    def _calculate_reward_and_coverage(self):  # Tính phần thưởng và tỷ lệ bao phủ
        self.covered_heatmap = np.zeros(self.heatmap_shape, dtype=np.float32)  # Đặt lại vùng bao phủ
        heatmap = self.heatmap_vector.reshape(self.heatmap_shape)  # Chuyển heatmap thành 2D
        
        # Cập nhật covered_heatmap từ tất cả slices
        for x, y, w, h in self.slices:  # Duyệt qua từng slice
            self.covered_heatmap[y:y+h, x:x+w] = np.maximum(
                self.covered_heatmap[y:y+h, x:x+w], heatmap[y:y+h, x:x+w]
            )  # Lấy giá trị lớn nhất giữa vùng hiện tại và heatmap
        
        # Tính điểm thưởng dựa trên mức độ đậm của pixel
        reward = 0
        for i in range(self.height):  # Duyệt qua từng pixel
            for j in range(self.width):
                if self.covered_heatmap[i, j] > 0:  # Nếu pixel được bao phủ
                    val = heatmap[i, j]  # Lấy giá trị heatmap tại pixel
                    if val >= 0.75:  # Đậm nhất
                        reward += 2
                    elif val >= 0.5:  # Đậm vừa
                        reward += 1
                    elif val >= 0.25:  # Nhạt vừa
                        reward -= 1
                    else:  # Nhạt nhất
                        reward -= 2
        
        # Tính tỷ lệ bao phủ vùng quan trọng
        total_important = np.sum(heatmap >= 0.5)  # Tổng giá trị vùng quan trọng
        covered_important = np.sum(self.covered_heatmap * (heatmap >= 0.5))  # Tổng giá trị vùng quan trọng được bao phủ
        coverage_ratio = covered_important / total_important if total_important > 0 else 0  # Tỷ lệ bao phủ
        
        # Phạt số slice để tối ưu hóa
        reward -= 0.5 * len(self.slices)  # Phạt 0.5 cho mỗi slice
        
        return reward, coverage_ratio  # Trả về phần thưởng và tỷ lệ bao phủ

    def _optimize_slices(self):  # Tối ưu số slice bằng cách loại bỏ overlap
        heatmap = self.heatmap_vector.reshape(self.heatmap_shape)  # Chuyển heatmap thành 2D
        i = 0
        while i < len(self.slices):  # Duyệt qua từng slice
            x1, y1, w1, h1 = self.slices[i]  # Lấy thông tin slice thứ i
            region1 = heatmap[y1:y1+h1, x1:x1+w1]  # Vùng heatmap của slice i
            score1 = np.sum(region1[region1 >= 0.5])  # Điểm bao phủ vùng quan trọng của slice i
            
            j = i + 1
            while j < len(self.slices):  # So sánh với các slice sau
                x2, y2, w2, h2 = self.slices[j]  # Lấy thông tin slice thứ j
                overlap_x = max(0, min(x1+w1, x2+w2) - max(x1, x2))  # Tính độ overlap theo trục x
                overlap_y = max(0, min(y1+h1, y2+h2) - max(y1, y2))  # Tính độ overlap theo trục y
                overlap_area = overlap_x * overlap_y  # Diện tích overlap
                area1 = w1 * h1  # Diện tích slice i
                if overlap_area > 0.5 * area1:  # Nếu overlap > 50% diện tích slice i
                    region2 = heatmap[y2:y2+h2, x2:x2+w2]  # Vùng heatmap của slice j
                    score2 = np.sum(region2[region2 >= 0.5])  # Điểm bao phủ vùng quan trọng của slice j
                    if score1 > score2:  # Nếu slice i tốt hơn
                        del self.slices[j]  # Xóa slice j
                    else:  # Nếu slice j tốt hơn
                        del self.slices[i]  # Xóa slice i
                        i -= 1  # Giảm i vì danh sách bị thay đổi
                        break
                else:
                    j += 1  # Không overlap đáng kể, chuyển sang slice tiếp theo
            i += 1  # Chuyển sang slice tiếp theo

    def step(self, action):  # Thực hiện một bước trong môi trường
        self.current_step += 1  # Tăng số bước hiện tại
        create_flag, slice_idx, x_ratio, y_ratio, w_ratio, h_ratio = action  # Tách action thành các thành phần
        
        heatmap = self.heatmap_vector.reshape(self.heatmap_shape)  # Chuyển heatmap thành 2D
        
        if create_flag >= 0.5:  # Nếu agent muốn tạo slice mới
            x = int(x_ratio * (self.width - 10))  # Tính tọa độ x từ tỷ lệ
            y = int(y_ratio * (self.height - 10))  # Tính tọa độ y từ tỷ lệ
            w = max(10, int(w_ratio * self.width * 0.5))  # Tính chiều rộng (tối thiểu 10)
            h = max(10, int(h_ratio * self.height * 0.5))  # Tính chiều cao (tối thiểu 10)
            w = min(w, self.width - x)  # Giới hạn chiều rộng không vượt quá heatmap
            h = min(h, self.height - y)  # Giới hạn chiều cao không vượt quá heatmap
            
            if len(self.slices) < self.max_slices:  # Nếu số slice chưa đạt tối đa
                self.slices.append((x, y, w, h))  # Thêm slice mới vào danh sách
            else:  # Nếu đã đạt max_slices, thay thế slice kém nhất
                scores = [np.sum(heatmap[y:y+h, x:x+w] * (heatmap[y:y+h, x:x+w] >= 0.5)) for x, y, w, h in self.slices]  # Tính điểm cho từng slice hiện có
                worst_idx = np.argmin(scores)  # Tìm chỉ số slice có điểm thấp nhất
                self.slices[worst_idx] = (x, y, w, h)  # Thay thế slice kém nhất bằng slice mới
        
        elif len(self.slices) > 0:  # Nếu tinh chỉnh slice hiện có và có slice
            slice_idx = min(int(slice_idx), len(self.slices) - 1)  # Chọn slice hợp lệ
            x, y, w, h = self.slices[slice_idx]  # Lấy thông tin slice
            
            # Tinh chỉnh 4 phía
            x_new = max(0, x + int(w_ratio * self.width))  # Điều chỉnh bên trái (w_ratio làm left_adj)
            w_new = max(10, w - int(w_ratio * self.width) + int(h_ratio * self.height))  # Điều chỉnh chiều rộng (h_ratio làm right_adj)
            y_new = max(0, y + int(x_ratio * self.height))  # Điều chỉnh phía trên (x_ratio làm top_adj)
            h_new = max(10, h - int(x_ratio * self.height) + int(y_ratio * self.width))  # Điều chỉnh chiều cao (y_ratio làm bottom_adj)
            
            w_new = min(w_new, self.width - x_new)  # Giới hạn chiều rộng
            h_new = min(h_new, self.height - y_new)  # Giới hạn chiều cao
            self.slices[slice_idx] = (x_new, y_new, w_new, h_new)  # Cập nhật slice
        
        # Tối ưu số slice
        self._optimize_slices()  # Gọi hàm loại bỏ slice overlap
        
        # Tính phần thưởng và tỷ lệ bao phủ
        reward, coverage_ratio = self._calculate_reward_and_coverage()
        done = coverage_ratio >= self.coverage_threshold or self.current_step >= self.max_steps  # Kiểm tra điều kiện kết thúc
        
        return self._get_state(), reward, done, {"heatmap_sum": reward, "coverage_ratio": coverage_ratio}  # Trả về trạng thái mới, reward, done, và info

    def render(self):  # Hiển thị heatmap và các slice
        plt.imshow(self.heatmap_rgb)  # Vẽ ảnh RGB gốc
        for x, y, w, h in self.slices:  # Duyệt qua từng slice
            plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='white', lw=2))  # Vẽ hình chữ nhật trắng không tô màu
        plt.show()  # Hiển thị hình

# --- BƯỚC 4: XÂY DỰNG DEEP Q-LEARNING AGENT ---
class DQNAgent:  # Lớp agent DQN
    class QNetwork(nn.Module):  # Mạng neural Q-network
        def __init__(self, state_size, action_size=6):  # Khởi tạo mạng với state_size và action_size=6
            super(DQNAgent.QNetwork, self).__init__()  # Gọi hàm khởi tạo của nn.Module
            self.fc1 = nn.Linear(state_size, 256)  # Lớp fully connected đầu tiên: state_size -> 256
            self.fc2 = nn.Linear(256, 128)  # Lớp thứ hai: 256 -> 128
            self.fc3 = nn.Linear(128, 64)  # Lớp thứ ba: 128 -> 64
            self.fc4 = nn.Linear(64, action_size)  # Lớp đầu ra: 64 -> action_size (6)
            self.relu = nn.ReLU()  # Hàm kích hoạt ReLU
            self.sigmoid = nn.Sigmoid()  # Hàm kích hoạt Sigmoid (cho create_flag)
        
        def forward(self, x):  # Lan truyền tiến
            x = self.relu(self.fc1(x))  # Qua lớp 1 và ReLU
            x = self.relu(self.fc2(x))  # Qua lớp 2 và ReLU
            x = self.relu(self.fc3(x))  # Qua lớp 3 và ReLU
            x = self.fc4(x)  # Qua lớp đầu ra
            x[:, 0] = self.sigmoid(x[:, 0])  # Áp dụng Sigmoid cho create_flag (cột 0) để ra [0, 1]
            return x  # Trả về action vector

    def __init__(self, state_size):  # Khởi tạo agent
        self.state_size = state_size  # Kích thước trạng thái
        self.action_size = 6  # Kích thước hành động (6 thành phần)
        self.replay_buffer = ReplayBuffer(max_size=2000)  # Khởi tạo replay buffer
        self.gamma = 0.95  # Hệ số giảm giá (discount factor)
        self.epsilon = 1.0  # Khởi tạo epsilon cho epsilon-greedy
        self.epsilon_min = 0.05  # Giá trị epsilon tối thiểu
        self.epsilon_decay = 0.99  # Tỷ lệ giảm epsilon sau mỗi episode
        self.learning_rate = 0.001  # Tốc độ học của optimizer

        self.q_network = self.QNetwork(state_size, self.action_size)  # Mạng Q chính
        self.target_network = self.QNetwork(state_size, self.action_size)  # Mạng target (bản sao của Q-network)
        self.update_target_network()  # Đồng bộ target network với Q-network
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)  # Optimizer Adam

    def update_target_network(self):  # Cập nhật target network
        self.target_network.load_state_dict(self.q_network.state_dict())  # Sao chép trọng số từ Q-network sang target

    def remember(self, state, action, reward, next_state, done):  # Lưu kinh nghiệm vào buffer
        self.replay_buffer.add(state, action, reward, next_state, done)  # Gọi phương thức add của ReplayBuffer

    def act(self, state):  # Chọn hành động dựa trên epsilon-greedy
        if np.random.rand() <= self.epsilon:  # Nếu ngẫu nhiên nhỏ hơn epsilon (khám phá)
            action = np.random.uniform([0, 0, 0, 0, -0.1, -0.1], [1, 10, 1, 1, 0.1, 0.1], 6)  # Tạo action ngẫu nhiên trong phạm vi action_space
            return action
        
        state = torch.FloatTensor(state).unsqueeze(0)  # Chuyển state thành tensor và thêm chiều batch
        self.q_network.eval()  # Chuyển Q-network sang chế độ đánh giá (không huấn luyện)
        with torch.no_grad():  # Tắt tính gradient để tiết kiệm tài nguyên
            action_values = self.q_network(state)  # Dự đoán action từ Q-network
        self.q_network.train()  # Chuyển lại Q-network sang chế độ huấn luyện
        return action_values.squeeze(0).numpy()  # Chuyển tensor thành numpy array và trả về
    
    def replay(self, batch_size):  # Huấn luyện agent từ replay buffer
        batch = self.replay_buffer.sample(batch_size)  # Lấy mẫu batch từ buffer
        if batch is None:  # Nếu không đủ dữ liệu
            return  # Thoát hàm
        states, actions, rewards, next_states, dones = batch  # Tách batch thành các thành phần
        states = torch.FloatTensor(states)  # Chuyển states thành tensor
        actions = torch.FloatTensor(actions)  # Chuyển actions thành tensor
        rewards = torch.FloatTensor(rewards).unsqueeze(1)  # Chuyển rewards thành tensor và thêm chiều
        next_states = torch.FloatTensor(next_states)  # Chuyển next_states thành tensor
        dones = torch.FloatTensor(dones).unsqueeze(1)  # Chuyển dones thành tensor và thêm chiều
        
        self.target_network.eval()  # Chuyển target network sang chế độ đánh giá
        with torch.no_grad():  # Tắt gradient
            next_q_values = self.target_network(next_states)  # Dự đoán Q-values từ target network
            next_q_value = torch.max(next_q_values, dim=1)[0].unsqueeze(1)  # Lấy giá trị max theo action và thêm chiều
            targets = rewards + self.gamma * next_q_value * (1 - dones)  # Tính target Q-values (Bellman equation)
        
        self.q_network.train()  # Chuyển Q-network sang chế độ huấn luyện
        q_values = self.q_network(states)  # Dự đoán Q-values từ Q-network
        loss = nn.MSELoss()(q_values, torch.cat([targets, actions[:, 1:]], dim=1))  # Tính loss (MSE giữa Q-values và target)
        self.optimizer.zero_grad()  # Xóa gradient cũ
        loss.backward()  # Tính gradient
        self.optimizer.step()  # Cập nhật trọng số mạng

# --- BƯỚC 5: HUẤN LUYỆN ---
def train_dqn(heatmap_path, episodes=1000, batch_size=64):  # Hàm huấn luyện DQN
    heatmap_vector, heatmap_rgb = preprocess_heatmap(heatmap_path)  # Tiền xử lý heatmap
    heatmap_shape = (heatmap_rgb.shape[0], heatmap_rgb.shape[1])  # Lấy kích thước heatmap
    env = SlicingEnvironment(heatmap_vector, heatmap_shape, heatmap_rgb, coverage_threshold=0.9, max_steps=50)  # Khởi tạo môi trường
    agent = DQNAgent(env.observation_space.shape[0])  # Khởi tạo agent với state_size
    
    rewards = []  # Danh sách lưu tổng reward mỗi episode
    best_reward = float('-inf')  # Khởi tạo reward tốt nhất là âm vô cực
    best_params = None  # Biến lưu tham số tốt nhất của Q-network
    
    for episode in range(episodes):  # Vòng lặp qua số episode
        # Tải tham số tốt nhất từ episode trước (nếu có)
        if best_params is not None:  # Nếu đã có tham số tốt nhất
            agent.q_network.load_state_dict(best_params)  # Tải tham số tốt nhất vào Q-network
            agent.update_target_network()  # Đồng bộ target network với Q-network
        
        state = env.reset()  # Đặt lại môi trường
        total_reward = 0  # Tổng reward trong episode
        for step in range(50):  # Vòng lặp tối đa 50 bước trong episode
            action = agent.act(state)  # Agent chọn hành động
            next_state, reward, done, info = env.step(action)  # Thực hiện hành động, nhận phản hồi từ môi trường
            agent.remember(state, action, reward, next_state, done)  # Lưu kinh nghiệm vào buffer
            agent.replay(batch_size)  # Huấn luyện từ buffer
            state = next_state  # Cập nhật trạng thái
            total_reward += reward  # Cộng dồn reward
            if done:  # Nếu episode kết thúc
                break  # Thoát vòng lặp
        
        # Cập nhật tham số tốt nhất nếu episode này tốt hơn
        if total_reward > best_reward:  # Nếu reward hiện tại vượt qua best_reward
            best_reward = total_reward  # Cập nhật best_reward
            best_params = agent.q_network.state_dict()  # Lưu trạng thái của Q-network
        
        if agent.epsilon > agent.epsilon_min:  # Nếu epsilon chưa đạt tối thiểu
            agent.epsilon *= agent.epsilon_decay  # Giảm epsilon (giảm khám phá)
        rewards.append(total_reward)  # Lưu tổng reward
        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward:.2f}, Best Reward: {best_reward:.2f}, Epsilon: {agent.epsilon:.3f}, Coverage: {info['coverage_ratio']:.3f}, Slices: {len(env.slices)}")  # In thông tin episode
        if episode % 20 == 0:  # Cứ sau 20 episode
            agent.update_target_network()  # Cập nhật target network
            env.render()  # Hiển thị kết quả
            plt.plot(rewards)  # Vẽ đồ thị reward
            plt.xlabel("Episode")  # Nhãn trục x
            plt.ylabel("Total Reward")  # Nhãn trục y
            plt.show()  # Hiển thị đồ thị

if __name__ == "__main__":  # Điểm vào chương trình
    heatmap_path = "/content/drive/MyDrive/Project/[RL_OD] - RL for object detection/Test/heatmap_dog_background_1.jpg"  # Đường dẫn đến heatmap
    train_dqn(heatmap_path, episodes=1000)  # Chạy huấn luyện với 1000 episode