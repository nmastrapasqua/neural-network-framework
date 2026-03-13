#include "training_monitor.h"
#include <iostream>
#include <iomanip>
#include <numeric>

TrainingMonitor::TrainingMonitor(size_t print_interval)
    : print_interval_(print_interval) {
}

void TrainingMonitor::recordEpoch(size_t /* epoch */, double loss, double accuracy) {
    loss_history_.push_back(loss);
    accuracy_history_.push_back(accuracy);
}

void TrainingMonitor::printProgress(size_t epoch, size_t total_epochs) const {
    if (loss_history_.empty()) {
        return;
    }

    // Only print at specified intervals, first epoch, and last epoch
    bool should_print = (epoch == 0) ||
                       ((epoch + 1) % print_interval_ == 0) ||
                       (epoch + 1 == total_epochs);

    if (!should_print) {
    	return;
    }

    // Get the most recent metrics
    double current_loss = loss_history_.back();
    double current_accuracy = accuracy_history_.back();

    // Calculate progress percentage
    double progress = (static_cast<double>(epoch + 1) / total_epochs) * 100.0;

    // Print formatted progress
    std::cout << "Epoch " << std::setw(4) << (epoch + 1)
              << "/" << total_epochs
              << " [" << std::fixed << std::setprecision(1) << progress << "%]"
              << " - Loss: " << std::setprecision(6) << current_loss
              << " - Accuracy: " << std::setprecision(4) << (current_accuracy * 100.0) << "%"
              << std::endl;
}

double TrainingMonitor::getAverageLoss() const {
    if (loss_history_.empty()) {
        return 0.0;
    }

    double sum = std::accumulate(loss_history_.begin(), loss_history_.end(), 0.0);
    return sum / loss_history_.size();
}

const std::vector<double>& TrainingMonitor::getLossHistory() const {
    return loss_history_;
}

const std::vector<double>& TrainingMonitor::getAccuracyHistory() const {
    return accuracy_history_;
}
