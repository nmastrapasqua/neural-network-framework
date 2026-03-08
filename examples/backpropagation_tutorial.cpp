#include "layer.h"
#include "activation.h"
#include "network.h"
#include "loss.h"
#include <iostream>
#include <memory>

/*
 * Tutorial Backpropagation: Esempio Completo Passo per Passo
 * Per verificare velocemente i calcoli si può usare questi solver:
 * 	- https://it.symbolab.com/solver/rational-expression-calculator/
 * 	- https://www.mathcelebrity.com/sigmoid-function-calculator.php
 */
void initLayers(Layer& layer_1, Layer& layer_2, Layer& layer_3) {
	std::cout << "Inizializzazione dei pesi e biases Layer 1" << std::endl;
	Matrix& W1 = layer_1.getWeights();
	W1(0 ,0) = 0.1; W1(0 ,1) = 0.2;
	W1(1 ,0) = 0.3; W1(1 ,1) = 0.4;
	W1(2 ,0) = 0.5; W1(2 ,1) = 0.6;
	W1.print("W1");
	Vector& b1 = layer_1.getBiases();
	b1[0] = 0.3; b1[1] = 0.4; b1[2] = 0.5;
	b1.print("b1");

	std::cout << std::endl << "Inizializzazione dei pesi e biases Layer 2" << std::endl;
	Matrix& W2 = layer_2.getWeights();
	W2(0 ,0) = 0.7; W2(0 ,1) = 0.8; W2(0 ,2) = 0.9;
	W2(1 ,0) = 0.2; W2(1 ,1) = 0.3; W2(1 ,2) = 0.4;
	W2.print("W2");
	Vector& b2 = layer_2.getBiases();
	b2[0] = 0.1; b2[1] = 0.2;
	b2.print("b2");

	std::cout << std::endl << "Inizializzazione dei pesi e biases Layer 3" << std::endl;
	Matrix& W3 = layer_3.getWeights();
	W3(0 ,0) = 0.5; W3(0 ,1) = 0.6;
	W3.print("W3");
	Vector& b3 = layer_3.getBiases();
	b3[0] = 0.1;
	b3.print("b3");
}

int main() {
	std::cout << "=============================" << std::endl;
	 std::cout << "Tutorial Backpropagation" << std::endl;
	 std::cout << "=============================" << std::endl;

	 MeanSquaredError loss;
	 auto sigmoid = std::make_shared<Sigmoid>();
	 std::cout << "Loss: " << loss.name() << ", Activation: "<< sigmoid->name() <<  std::endl;

	 std::cout << "Network architecture: [2, 3, 2, 1]" << std::endl << std::endl;
	 Layer l1(2, 3, sigmoid);
	 Layer l2(3, 2, sigmoid);
	 Layer l3(2, 1, sigmoid);
	 initLayers(l1, l2, l3);
	 Matrix& W1 = l1.getWeights();
	 Vector& b1 = l1.getBiases();
	 Matrix& W2 = l2.getWeights();
	 Vector& b2 = l2.getBiases();
	 Matrix& W3 = l3.getWeights();
	 Vector& b3 = l3.getBiases();

	 std::cout << "Input e target" << std::endl;
	 Vector input{0.5, 0.8};
	 Vector target{1.0};
	 input.print("x_input");
	 target.print("y_target");
	 std::vector<Vector> inputs = {input};
	 std::vector<Vector> targets = {target};
	 std::cout << std::endl;

	 std::cout << "FASE 1: FORWARD PROPAGATION" << std::endl;
	 std::cout << "=============================" << std::endl;
	 std::cout << "Step 1: Input Layer → Hidden Layer 1" << std::endl;
	 std::cout << "-----------------------------" << std::endl;
	 std::cout << "Calcolo z¹ = W¹ · a⁰ + b¹" << std::endl;
	 std::cout << "z¹₁ = 0.1×0.5 + 0.2×0.8 + 0.3 = 0.51" << std::endl;
	 std::cout << "z¹₂ = 0.3×0.5 + 0.4×0.8 + 0.4 = 0.87" << std::endl;
	 std::cout << "z¹₃ = 0.5×0.5 + 0.6×0.8 + 0.5 = 1.23" << std::endl;
	 Vector z1 = W1*input + b1;
	 z1.print("z¹");
	 std::cout << "Calcolo a¹ = σ(z¹)" << std::endl;
	 std::cout << "a¹₁ = σ(0.51) = 0.624806" << std::endl;
	 std::cout << "a¹₂ = σ(0.87) = 0.704746" << std::endl;
	 std::cout << "a¹₃ = σ(1.23) = 0.773819" << std::endl;
	 Vector a1{sigmoid->activate(z1[0]), sigmoid->activate(z1[1]), sigmoid->activate(z1[2])};
	 a1.print("a¹");

	 std::cout << std::endl << "Step 2: Hidden Layer 1 → Hidden Layer 2" << std::endl;
	 std::cout << "-----------------------------" << std::endl;
	 std::cout << "Calcolo z² = W² · a¹ + b²" << std::endl;
	 std::cout << "z²₁ = 0.7×0.624806 + 0.8×0.704746 + 0.9×0.773819 + 0.1 = 1.7976" << std::endl;
	 std::cout << "z²₂ = 0.2×0.624806 + 0.3×0.704746 + 0.4×0.773819 + 0.2 = 0.8459" << std::endl;
	 Vector z2 = W2*a1 + b2;
	 z2.print("z²");
	 std::cout << "Calcolo a² = σ(z²)" << std::endl;
	 std::cout << "a²₁ = σ(1.7976) = 0.8579" << std::endl;
	 std::cout << "a²₂ = σ(0.8459) = 0.6997" << std::endl;
	 Vector a2{sigmoid->activate(z2[0]), sigmoid->activate(z2[1])};
	 a2.print("a²");

	 std::cout << std::endl << "Step 3: Hidden Layer 2 → Output Layer" << std::endl;
	 std::cout << "-----------------------------" << std::endl;
	 std::cout << "Calcolo z³ = W³ · a² + b³" << std::endl;
	 std::cout << "z³ = 0.5×0.8579 + 0.6×0.6997 + 0.1 = 0.9488" << std::endl;
	 Vector z3 = W3*a2 + b3;
	 z3.print("z³");
	 std::cout << "Calcolo a³ = σ(z³)" << std::endl;
	 std::cout << "a³ = σ(0.9488) = 0.7209" << std::endl;
	 Vector a3{sigmoid->activate(z3[0])};
	 a3.print("y_pred = a³");

	 std::cout << std::endl << "Step 4: Calcolo della Loss" << std::endl;
	 std::cout << "-----------------------------" << std::endl;
	 std::cout << "MSE Loss = (y_pred - y_target)² =  (0.7209 - 1.0)² = 0.0779" << std::endl;
	 std::cout << "L = " << loss.compute(a3, target) << std::endl;

	 std::cout << std::endl << "FASE 2: BACKWARD PROPAGATION" << std::endl;
	 std::cout << "=============================" << std::endl;
	 std::cout << "Ora calcoliamo i gradienti per aggiornare i pesi. Procediamo a ritroso." << std::endl;
	 std::cout << "Derivate fondamentali per la backpropagation:" << std::endl;
	 std::cout << "\tsigmoid: σ'(z) = σ(z) × (1 - σ(z))" << std::endl;
	 std::cout << "\tMSE: ∂L/∂y_pred = (2/n) × (y_pred - y_target)" << std::endl;
	 std::cout << "\t\tsicalcola con la chain rule:" << std::endl;
	 std::cout << "\t\t∂L/∂y_pred = ∂L/∂u × ∂u/∂y_pred = (2/n) × u × 1 = (2/n) × (y_pred - y_target)" << std::endl;
	 std::cout << "\t\tcon u = y_pred - y_target" << std::endl;
	 std::cout << "\tSomma pesata: z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b" << std::endl;
	 std::cout << "\t\tad esempio ∂z/∂w₁ = x₁" << std::endl;

	 std::cout << std::endl << "Step 1: Calcolo δ³ (gradiente della loss rispetto alla pre-attivazione z³)" << std::endl;
	 std::cout << "-----------------------------" << std::endl;
	 Vector derivataLoss3 = loss.gradient(a3, target);
	 derivataLoss3.print("∂L/∂y_pred = ∂L/∂a³");
	 Vector derivataSigmoid3{sigmoid->derivative(z3[0])};
	 derivataSigmoid3.print("∂a³/∂z³");
	 Vector delta3{derivataLoss3[0] * derivataSigmoid3[0]};
	 delta3.print("δ³ = ∂L/∂z³ = ∂L/∂a³ × ∂a³/∂z³");


	 return 0;
}



