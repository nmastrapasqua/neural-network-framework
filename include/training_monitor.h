#ifndef TRAINING_MONITOR_H
#define TRAINING_MONITOR_H

#include <vector>
#include <cstddef>

/**
 * TrainingMonitor - Tracks and displays training progress
 *
 * This class records loss and accuracy metrics during training,
 * provides progress feedback to the user, and maintains history
 * for analysis and visualization.
 */
class TrainingMonitor {
public:
    /**
     * Record metrics for a completed epoch
     * @param epoch The epoch number (0-indexed)
     * @param loss The average loss for this epoch
     * @param accuracy The accuracy for this epoch (0.0 to 1.0)
     */
    void recordEpoch(size_t epoch, double loss, double accuracy);

    /**
     * Print training progress to console
     * @param epoch Current epoch number (0-indexed)
     * @param total_epochs Total number of epochs in training
     */
    void printProgress(size_t epoch, size_t total_epochs) const;

    /**
     * Calculate average loss across all recorded epochs
     * @return Average loss, or 0.0 if no epochs recorded
     */
    double getAverageLoss() const;

    /**
     * Get the complete loss history
     * @return Vector of loss values, one per epoch
     */
    const std::vector<double>& getLossHistory() const;

    /**
     * Get the complete accuracy history
     * @return Vector of accuracy values, one per epoch
     */
    const std::vector<double>& getAccuracyHistory() const;

private:
    std::vector<double> loss_history_;
    std::vector<double> accuracy_history_;
};

#endif // TRAINING_MONITOR_H
