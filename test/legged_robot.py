#ifndef RL_TOOLS_RL_ENVIRONMENTS_MUJOCO_A1_OPERATIONS_CPU_H
#define RL_TOOLS_RL_ENVIRONMENTS_MUJOCO_A1_OPERATIONS_CPU_H

#include "a1.h"
#include "../../operations_generic.h"
#include <cstring>
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::mujoco::a1 {
    #include "model.h"
}

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    // Allocate memory for the A1 environment
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::environments::mujoco::A1<SPEC>& env) {
        using TI = typename DEVICE::index_t;
        constexpr typename DEVICE::index_t error_length = 1000;
        char error[error_length] = "Could not load model";

        // Load the model from embedded XML
        {
            mjVFS* vfs = new mjVFS;
            mj_defaultVFS(vfs);
            mj_makeEmptyFileVFS(vfs, "model.xml", rl_tools::rl::environments::mujoco::a1::model_xml_len);
            int file_idx = mj_findFileVFS(vfs, "model.xml");
            std::memcpy(vfs->filedata[file_idx], rl_tools::rl::environments::mujoco::a1::model_xml, rl_tools::rl::environments::mujoco::a1::model_xml_len);
            env.model = mj_loadXML("model.xml", vfs, error, error_length);
            mj_deleteFileVFS(vfs, "model.xml");
            delete vfs;
        }
#ifdef RL_TOOLS_DEBUG_RL_ENVIRONMENTS_MUJOCO_CHECK_INIT
        utils::assert_exit(device, env.model != nullptr, error);
#endif
        env.data = mj_makeData(env.model);

        // Initialize state variables
        for (TI state_i = 0; state_i < SPEC::STATE_DIM_Q; state_i++) {
            env.init_q[state_i] = env.data->qpos[state_i];
        }
        for (TI state_i = 0; state_i < SPEC::STATE_DIM_Q_DOT; state_i++) {
            env.init_q_dot[state_i] = env.data->qvel[state_i];
        }

        env.torso_id = mj_name2id(env.model, mjOBJ_BODY, "trunk");
    }

    // Free memory for the A1 environment
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::environments::mujoco::A1<SPEC>& env) {
        mj_deleteData(env.data);
        mj_deleteModel(env.model);
    }

    // Initialize the A1 environment
    template <typename DEVICE, typename SPEC>
    void init(DEVICE& device, rl::environments::mujoco::A1<SPEC>& env, typename rl::environments::mujoco::A1<SPEC>::Parameters& parameters) {
        // No special initialization required here
    }

    // Reset state with noise
    template <typename DEVICE, typename SPEC, typename RNG>
    void sample_initial_state(DEVICE& device, const rl::environments::mujoco::A1<SPEC>& env, typename rl::environments::mujoco::A1<SPEC>::Parameters& parameters, typename rl::environments::mujoco::a1::State<SPEC>& state, RNG& rng) {
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;

        mj_resetData(env.model, env.data);

        for (TI state_i = 0; state_i < SPEC::STATE_DIM_Q; state_i++) {
            state.q[state_i] = env.init_q[state_i] + random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), -SPEC::PARAMETERS::RESET_NOISE_SCALE, SPEC::PARAMETERS::RESET_NOISE_SCALE, rng);
        }
        for (TI state_i = 0; state_i < SPEC::STATE_DIM_Q_DOT; state_i++) {
            state.q_dot[state_i] = env.init_q_dot[state_i] + random::normal_distribution::sample(typename DEVICE::SPEC::RANDOM(), (T)0, SPEC::PARAMETERS::RESET_NOISE_SCALE, rng);
        }

        mj_forward(env.model, env.data);
    }

    // Environment step function
    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    typename SPEC::T step(DEVICE& device, rl::environments::mujoco::A1<SPEC>& env, typename rl::environments::mujoco::A1<SPEC>::Parameters& parameters, const rl::environments::mujoco::a1::State<SPEC>& state, const Matrix<ACTION_SPEC>& action, rl::environments::mujoco::a1::State<SPEC>& next_state, RNG& rng) {
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        using ENVIRONMENT = rl::environments::mujoco::A1<SPEC>;

        static_assert(ACTION_SPEC::ROWS == 1);
        static_assert(ACTION_SPEC::COLS == ENVIRONMENT::ACTION_DIM);

        T control_cost = 0;

        // Apply actions and compute control cost
        for (TI action_i = 0; action_i < SPEC::ACTION_DIM; action_i++) {
            T control = get(action, 0, action_i);
            control = math::clamp<T>(device.math, control, -1, 1);
            env.data->ctrl[action_i] = control;
            control_cost += control * control;
        }
        control_cost *= SPEC::PARAMETERS::CONTROL_COST_WEIGHT;

        // Set state
        for (TI state_i = 0; state_i < SPEC::STATE_DIM_Q; state_i++) {
            env.data->qpos[state_i] = state.q[state_i];
        }
        for (TI state_i = 0; state_i < SPEC::STATE_DIM_Q_DOT; state_i++) {
            env.data->qvel[state_i] = state.q_dot[state_i];
        }

        // Simulate frames
        for (TI frame_i = 0; frame_i < SPEC::PARAMETERS::FRAME_SKIP; frame_i++) {
            mj_step(env.model, env.data);
        }

        // Get next state
        for (TI state_i = 0; state_i < SPEC::STATE_DIM_Q; state_i++) {
            next_state.q[state_i] = env.data->qpos[state_i];
        }
        for (TI state_i = 0; state_i < SPEC::STATE_DIM_Q_DOT; state_i++) {
            next_state.q_dot[state_i] = env.data->qvel[state_i];
        }

        // Calculate rewards
        T healthy_reward = next_state.q[2] >= SPEC::PARAMETERS::HEALTHY_Z_MIN && next_state.q[2] <= SPEC::PARAMETERS::HEALTHY_Z_MAX ? SPEC::PARAMETERS::HEALTHY_REWARD : 0;
        T velocity_reward = env.data->qvel[0] * SPEC::PARAMETERS::VELOCITY_TRACKING_WEIGHT;
        env.last_reward = healthy_reward + velocity_reward - control_cost;

        // Check termination
        env.last_terminated = next_state.q[2] < SPEC::PARAMETERS::HEALTHY_Z_MIN || next_state.q[2] > SPEC::PARAMETERS::HEALTHY_Z_MAX;
        return SPEC::PARAMETERS::DT * SPEC::PARAMETERS::FRAME_SKIP;
    }

    // Observe the environment
    template <typename DEVICE, typename SPEC, typename OBS_TYPE_SPEC, typename OBS_SPEC, typename RNG>
    void observe(DEVICE& device, const rl::environments::mujoco::A1<SPEC>& env, const typename rl::environments::mujoco::A1<SPEC>::Parameters& parameters, const rl::environments::mujoco::a1::State<SPEC>& state, const rl::environments::mujoco::a1::Observation<OBS_TYPE_SPEC>&, Matrix<OBS_SPEC>& observation, RNG& rng) {
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;

        TI offset = 0;

        // Base velocities
        for (TI i = 0; i < 6; i++) {  // Linear (3) + Angular (3)
            set(observation, 0, offset++, env.data->qvel[i]);
        }

        // Projected gravity
        T gravity[3] = {0.0, 0.0, -9.8};
        for (TI i = 0; i < 3; i++) {
            set(observation, 0, offset++, gravity[i]);
        }

        // Joint positions and velocities
        for (TI i = 7; i < SPEC::STATE_DIM_Q; i++) {
            set(observation, 0, offset++, state.q[i]);
        }
        for (TI i = 6; i < SPEC::STATE_DIM_Q_DOT; i++) {
            set(observation, 0, offset++, state.q_dot[i]);
        }
    }

    // Return whether the episode is terminated
    template <typename DEVICE, typename SPEC, typename RNG>
    bool terminated(DEVICE& device, const rl::environments::mujoco::A1<SPEC>& env, const typename rl::environments::mujoco::A1<SPEC>::Parameters& parameters, const typename rl::environments::mujoco::a1::State<SPEC>& state, RNG& rng) {
        return env.last_terminated;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif