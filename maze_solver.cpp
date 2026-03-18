#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <deque>
#include <memory>
#include <chrono>
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

using namespace std;

// --- Neural Network Definitions ---
struct Layer {
    int in_size, out_size;
    vector<vector<double>> weights;
    vector<vector<double>> v_weights; // SGD Momentum
    vector<double> biases;
    vector<double> v_biases; // SGD Momentum
    vector<double> inputs;
    vector<double> outputs;
    vector<double> deltas;

    Layer(int in, int out, mt19937& gen) : in_size(in), out_size(out) {
        // He Initialization
        normal_distribution<double> dist(0.0, sqrt(2.0 / in));
        weights.assign(out, vector<double>(in, 0.0));
        v_weights.assign(out, vector<double>(in, 0.0));
        biases.assign(out, 0.0);
        v_biases.assign(out, 0.0);
        for (int i = 0; i < out; ++i) {
            for (int j = 0; j < in; ++j) {
                weights[i][j] = dist(gen);
            }
        }
        inputs.assign(in, 0.0);
        outputs.assign(out, 0.0);
        deltas.assign(out, 0.0);
    }
    
    void forward(const vector<double>& in_data, bool is_output) {
        inputs = in_data;
        for (int i = 0; i < out_size; ++i) {
            double sum = biases[i];
            for (int j = 0; j < in_size; ++j) {
                sum += weights[i][j] * inputs[j];
            }
            outputs[i] = is_output ? sum : max(0.0, sum); // ReLU activation
        }
    }
};

class NeuralNetwork {
public:
    vector<Layer> layers;
    double learning_rate;

    NeuralNetwork() : learning_rate(0.001) {} 

    NeuralNetwork(vector<int> topology, double lr, mt19937& gen) : learning_rate(lr) {
        for (size_t i = 1; i < topology.size(); ++i) {
            layers.emplace_back(topology[i-1], topology[i], gen);
        }
    }

    vector<double> predict(const vector<double>& input) {
        vector<double> cur_out = input;
        for (size_t i = 0; i < layers.size(); ++i) {
            layers[i].forward(cur_out, i == layers.size() - 1);
            cur_out = layers[i].outputs;
        }
        return cur_out;
    }

    void train(const vector<double>& input, const vector<double>& target) {
        predict(input); 

        // Output layer deltas
        for (int i = 0; i < layers.back().out_size; ++i) {
            layers.back().deltas[i] = layers.back().outputs[i] - target[i];
        }

        // Hidden layers backpropagation
        for (int l = layers.size() - 2; l >= 0; --l) {
            for (int i = 0; i < layers[l].out_size; ++i) {
                double error = 0.0;
                for (int j = 0; j < layers[l+1].out_size; ++j) {
                    error += layers[l+1].deltas[j] * layers[l+1].weights[j][i];
                }
                layers[l].deltas[i] = error * (layers[l].outputs[i] > 0 ? 1.0 : 0.0);
            }
        }

        // SGD Updates with Momentum
        double momentum = 0.9;
        for (size_t l = 0; l < layers.size(); ++l) {
            for (int i = 0; i < layers[l].out_size; ++i) {
                double grad_b = layers[l].deltas[i];
                layers[l].v_biases[i] = momentum * layers[l].v_biases[i] + learning_rate * grad_b;
                layers[l].biases[i] -= layers[l].v_biases[i];
                
                for (int j = 0; j < layers[l].in_size; ++j) {
                    double grad_w = layers[l].deltas[i] * layers[l].inputs[j];
                    layers[l].v_weights[i][j] = momentum * layers[l].v_weights[i][j] + learning_rate * grad_w;
                    layers[l].weights[i][j] -= layers[l].v_weights[i][j];
                }
            }
        }
    }
};

struct Experience {
    vector<double> state;
    int action;
    double reward;
    vector<double> next_state;
    bool done;
};

// --- Maze Definitions ---
class Maze {
public:
    int n, width;
    vector<int> grid;
    int start_x, start_y;
    int end_x, end_y;

    Maze(int size) : n(size), width(2 * size + 1) {
        grid.assign(width * width, 0); // 0 = wall
        start_x = 1; start_y = 1;
        end_x = width - 2; end_y = width - 2;
    }

    void generate(mt19937& gen) {
        vector<pair<int, int>> stack;
        stack.push_back({1, 1});
        grid[1 * width + 1] = 1;

        int dx[4] = {0, 0, 2, -2};
        int dy[4] = {2, -2, 0, 0};

        while (!stack.empty()) {
            auto [cx, cy] = stack.back();
            vector<int> dirs = {0, 1, 2, 3};
            shuffle(dirs.begin(), dirs.end(), gen);

            bool moved = false;
            for (int i : dirs) {
                int nx = cx + dx[i];
                int ny = cy + dy[i];

                if (nx > 0 && ny > 0 && nx < width - 1 && ny < width - 1 && grid[ny * width + nx] == 0) {
                    grid[ny * width + nx] = 1;
                    grid[(cy + dy[i]/2) * width + (cx + dx[i]/2)] = 1; // carve path between
                    stack.push_back({nx, ny});
                    moved = true;
                    break;
                }
            }
            if (!moved) {
                stack.pop_back();
            }
        }
        grid[end_y * width + end_x] = 1; 
    }

    bool is_valid(int x, int y) const {
        if (x < 0 || y < 0 || x >= width || y >= width) return false;
        return grid[y * width + x] == 1;
    }
};

// --- DQL Agent Definitions ---
class DQLAgent {
public:
    NeuralNetwork target_net;
    NeuralNetwork q_net;
    deque<Experience> memory;
    deque<Experience> goal_memory; // Prioritized buffer for sparse victories!
    int memory_size = 50000;
    int goal_memory_size = 1000;
    int batch_size = 64;
    double gamma = 0.99;
    double epsilon = 1.0;
    double epsilon_min = 0.05;
    double epsilon_decay = 0.999; // Slower decay for huge mazes
    int target_update_freq = 200; // Slower target sync for stability
    int steps = 0;
    mt19937& gen;

    DQLAgent(mt19937& g) : 
        // 12 inputs -> two 128 hidden layers -> 4 outputs
        q_net({12, 128, 128, 4}, 0.001, g), 
        target_net({12, 128, 128, 4}, 0.001, g), 
        gen(g) {
        target_net = q_net; 
    }

    vector<double> get_state(const Maze& maze, int x, int y, const vector<int>& visited) {
        vector<double> state(12, 0.0);
        state[0] = static_cast<double>(x) / maze.width;
        state[1] = static_cast<double>(y) / maze.width;
        state[2] = maze.is_valid(x, y - 1) ? 1.0 : 0.0;
        state[3] = maze.is_valid(x, y + 1) ? 1.0 : 0.0;
        state[4] = maze.is_valid(x - 1, y) ? 1.0 : 0.0;
        state[5] = maze.is_valid(x + 1, y) ? 1.0 : 0.0;
        state[6] = static_cast<double>(maze.end_x - x) / maze.width;
        state[7] = static_cast<double>(maze.end_y - y) / maze.width;
        
        // Memory of visited neighbors (allows agent to "see" if it's about to backtrack)
        state[8] = (y > 0 && visited[(y-1)*maze.width + x] > 0) ? 1.0 : 0.0;
        state[9] = (y < maze.width-1 && visited[(y+1)*maze.width + x] > 0) ? 1.0 : 0.0;
        state[10] = (x > 0 && visited[y*maze.width + x-1] > 0) ? 1.0 : 0.0;
        state[11] = (x < maze.width-1 && visited[y*maze.width + x+1] > 0) ? 1.0 : 0.0;

        return state;
    }

    int act(const vector<double>& state, bool exploring = true) {
        if (exploring) {
            uniform_real_distribution<double> dist(0.0, 1.0);
            if (dist(gen) < epsilon) {
                uniform_int_distribution<int> act_dist(0, 3);
                return act_dist(gen);
            }
        }
        auto q_values = q_net.predict(state);
        return distance(q_values.begin(), max_element(q_values.begin(), q_values.end()));
    }

    void remember(const vector<double>& state, int action, double reward, const vector<double>& next_state, bool done) {
        Experience exp = {state, action, reward, next_state, done};
        memory.push_back(exp);
        if (memory.size() > memory_size) memory.pop_front();
        
        // Push exclusively critical success states to the VIP goal memory
        if (reward == 100.0) {
            goal_memory.push_back(exp);
            if (goal_memory.size() > goal_memory_size) goal_memory.pop_front();
        }
    }

    void replay() {
        if (memory.size() < batch_size) return;
        
        vector<Experience> batch;
        sample(memory.begin(), memory.end(), back_inserter(batch), batch_size, gen);

        // Sparse Reward Injection mechanism -> Over-sample the goal transitions!
        if (!goal_memory.empty()) {
            uniform_int_distribution<size_t> dist(0, goal_memory.size() - 1);
            // Replace 4 slots of the 64-batch directly with vital goal paths
            for (int i = 0; i < min(4, (int)goal_memory.size()); ++i) {
                batch[i] = goal_memory[dist(gen)];
            }
        }

        for (const auto& exp : batch) {
            double target = exp.reward;
            if (!exp.done) {
                auto next_qs = target_net.predict(exp.next_state);
                target = exp.reward + gamma * (*max_element(next_qs.begin(), next_qs.end()));
            }
            auto target_f = q_net.predict(exp.state);
            target_f[exp.action] = target;
            q_net.train(exp.state, target_f);
        }

        if (epsilon > epsilon_min) epsilon *= epsilon_decay;

        steps++;
        if (steps % target_update_freq == 0) {
            target_net = q_net;
        }
    }
};

// --- SDL Visualization Details ---
void draw_rect(SDL_Renderer* renderer, int x, int y, int w, int h, int r, int g, int b, int a) {
    SDL_SetRenderDrawColor(renderer, r, g, b, a);
    SDL_FRect rect = {(float)x, (float)y, (float)w, (float)h};
    SDL_RenderFillRect(renderer, &rect);
}

int main(int argc, char* argv[]) {
    int maze_size = 5;
    cout << "Enter maze size (e.g. 5, max 20 recommended for fast learning): ";
    if (!(cin >> maze_size)) return 1;

    cout << "\n=============================================\n";
    cout << "       RL MAZE SOLVER VISUALIZER\n";
    cout << "=============================================\n";
    cout << "CONTROLS:\n";
    cout << "  [SPACE] - Toggle Fast-Forward Training Mode\n";
    cout << "=============================================\n\n";

    random_device rd;
    mt19937 gen(rd());

    Maze maze(maze_size);
    maze.generate(gen);

    DQLAgent agent(gen);

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        cerr << "SDL_Init Error: " << SDL_GetError() << endl;
        return 1;
    }

    int window_size = 800;
    SDL_Window* window = SDL_CreateWindow("RL Maze Solver", window_size, window_size, 0);
    if (!window) {
        cerr << "SDL_CreateWindow Error: " << SDL_GetError() << endl;
        SDL_Quit();
        return 1;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, NULL);
    if (!renderer) {
        cerr << "SDL_CreateRenderer Error: " << SDL_GetError() << endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    int cell_size = window_size / maze.width;

    bool running = true;
    SDL_Event event;

    int agent_x = maze.start_x;
    int agent_y = maze.start_y;
    int dx[4] = {0, 0, -1, 1}; // Up, Down, Left, Right
    int dy[4] = {-1, 1, 0, 0};

    bool training = true;
    bool fast_forward = false; // start slow so user sees it move!
    int episodes = 0;
    const int max_episodes = 2000;
    int steps_in_ep = 0;

    vector<pair<int, int>> current_path;
    vector<pair<int, int>> best_path;
    vector<int> explored(maze.width * maze.width, 0);
    vector<int> episode_visited(maze.width * maze.width, 0);

    current_path.push_back({agent_x, agent_y});

    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) {
                running = false;
            } else if (event.type == SDL_EVENT_KEY_DOWN) {
                if (event.key.key == SDLK_SPACE) {
                    fast_forward = !fast_forward;
                    if (fast_forward) cout << ">> Fast Forward Mode: ON (Accelerated Training)\n";
                    else cout << ">> Fast Forward Mode: OFF (Visualizing individual steps)\n";
                }
            }
        }

        // Single step if not fast forwarding! Prevents the "teleportation" feeling
        int steps_per_frame = training ? (fast_forward ? 100 : 1) : 1;

        for (int step = 0; step < steps_per_frame && running; ++step) {
            if (!training) {
                // Evaluation Mode
                vector<double> state = agent.get_state(maze, agent_x, agent_y, episode_visited);
                int action = agent.act(state, false); 
                
                int nx = agent_x + dx[action];
                int ny = agent_y + dy[action];
                
                if (maze.is_valid(nx, ny)) {
                    agent_x = nx;
                    agent_y = ny;
                    current_path.push_back({agent_x, agent_y});
                    episode_visited[agent_y * maze.width + agent_x]++;
                }
                
                if (agent_x == maze.end_x && agent_y == maze.end_y) {
                    agent_x = maze.start_x;
                    agent_y = maze.start_y;
                    best_path = current_path;
                    current_path.clear();
                    current_path.push_back({agent_x, agent_y});
                    fill(episode_visited.begin(), episode_visited.end(), 0);
                    episode_visited[agent_y * maze.width + agent_x]++;
                }
                
                SDL_Delay(50); // delay so it isn't instant
                break; // 1 step per frame always
            }

            // Training mode
            vector<double> state = agent.get_state(maze, agent_x, agent_y, episode_visited);
            int action = agent.act(state);

            int nx = agent_x + dx[action];
            int ny = agent_y + dy[action];

            double reward = 0.0;
            bool done = false;

            if (maze.is_valid(nx, ny)) {
                agent_x = nx;
                agent_y = ny;
                current_path.push_back({agent_x, agent_y});
                explored[agent_y * maze.width + agent_x] = 1;

                // Greatly improved reward structure prevents aimless wandering
                episode_visited[agent_y * maze.width + agent_x]++;
                if (episode_visited[agent_y * maze.width + agent_x] > 1) {
                    reward = -2.5; // Heavy Penalty for revisiting cells (looping)
                } else {
                    reward = 2.0;  // High Reward for discovering new geometry
                }

                if (agent_x == maze.end_x && agent_y == maze.end_y) {
                    reward = 100.0;
                    done = true;
                } else if (agent_x == maze.start_x && agent_y == maze.start_y) {
                    reward = -10.0; // Heavily penalize walking back to init
                }
            } else {
                reward = -5.0; // Penalty indicating wall hit
            }

            vector<double> next_state = agent.get_state(maze, agent_x, agent_y, episode_visited);
            agent.remember(state, action, reward, next_state, done);
            agent.replay();

            steps_in_ep++;
            // Terminate episode if won, or if agent takes way too many steps (stuck)
            if (done || steps_in_ep > maze.width * maze.width * 2) {
                if (done) {
                    cout << "Episode " << episodes << " won! Steps: " << steps_in_ep 
                         << " Epsilon: " << agent.epsilon << endl;
                    best_path = current_path;
                }
                
                // Reset for next episode
                agent_x = maze.start_x;
                agent_y = maze.start_y;
                steps_in_ep = 0;
                episodes++;
                
                current_path.clear();
                current_path.push_back({agent_x, agent_y});
                
                fill(episode_visited.begin(), episode_visited.end(), 0);

                if (episodes >= max_episodes) {
                    training = false;
                    agent_x = maze.start_x;
                    agent_y = maze.start_y;
                    current_path.clear();
                    current_path.push_back({agent_x, agent_y});
                    cout << "Training finished. Starting evaluation mode." << endl;
                }
                
                // Add a small visual delay on episode end during normal speed so it doesn't instantly snap back
                if (!fast_forward) SDL_Delay(200); 
            }
        }

        // --- RENDER ---
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        for (int y = 0; y < maze.width; ++y) {
            for (int x = 0; x < maze.width; ++x) {
                if (maze.grid[y * maze.width + x] == 0) {
                    draw_rect(renderer, x * cell_size, y * cell_size, cell_size, cell_size, 40, 40, 40, 255); // Wall
                } else {
                    if (!best_path.empty() && find(best_path.begin(), best_path.end(), make_pair(x, y)) != best_path.end()) {
                        draw_rect(renderer, x * cell_size, y * cell_size, cell_size, cell_size, 100, 255, 100, 255); // Green (Final Best)
                    } else if (explored[y * maze.width + x] == 1) {
                        draw_rect(renderer, x * cell_size, y * cell_size, cell_size, cell_size, 255, 100, 100, 255); // Red (Currently Explored subset)
                    } else {
                        draw_rect(renderer, x * cell_size, y * cell_size, cell_size, cell_size, 220, 220, 220, 255); // White (Unexplored)
                    }
                }
            }
        }

        // Draw start and end blocks clearly
        draw_rect(renderer, maze.start_x * cell_size, maze.start_y * cell_size, cell_size, cell_size, 0, 255, 255, 255); // Cyan
        draw_rect(renderer, maze.end_x * cell_size, maze.end_y * cell_size, cell_size, cell_size, 255, 0, 255, 255); // Magenta

        // Draw current path trail dynamically so user sees exact route agent takes (no teleporting visual!)
        if (current_path.size() > 1) {
            for (size_t i = 0; i < current_path.size() - 1; ++i) {
                int cx = current_path[i].first;
                int cy = current_path[i].second;
                draw_rect(renderer, cx * cell_size + cell_size/3, cy * cell_size + cell_size/3, cell_size/3, cell_size/3, 150, 150, 255, 255); // Light blue trail
            }
        }

        // Draw active Agent
        draw_rect(renderer, agent_x * cell_size + cell_size/4, agent_y * cell_size + cell_size/4, cell_size/2, cell_size/2, 50, 50, 255, 255); // Dark Blue

        SDL_RenderPresent(renderer);

        // Frame pacing logic
        if (!training) SDL_Delay(50);
        else if (!fast_forward) SDL_Delay(20); // Smooth real-time update
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
