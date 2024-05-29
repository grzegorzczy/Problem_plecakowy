import numpy as np
import sqlite3
import matplotlib.pyplot as plt

''' Variables '''
# Define knapsack problem parameters
max_weight = 80
max_space = 150
population_size = 50
# Always best answer at 5000, 99 centile at 100 
generations = 1000
mutation_rate = 0.02

# Connect to the database
conn = sqlite3.connect('items_list.db')
c = conn.cursor()

# Fetch items from the database
c.execute("SELECT weight, value, space FROM items")
items_data = c.fetchall()
conn.close()

''' Linking item data  '''
# Unpack items_data into weights and values lists
weights, values, spaces = zip(*items_data)

''' Functions for genetic algorithm '''
# Initialize population randomly
def initialize_population(population_size):
    # Initialize a population of size population_size with binary representation of items
    population = np.random.randint(2, size=(population_size, len(values)))
    return population.tolist()

# Calculate fitness of each individual
def calculate_fitness(individual):
    # Calculate the total weight, total space, and total value of the knapsack for an individual
    total_weight = np.sum(np.array(weights) * np.array(individual))
    total_space = np.sum(np.array(spaces) * np.array(individual))
    total_value = np.sum(np.array(values) * np.array(individual))
    # Check if the individual violates the weight or space constraints
    if total_weight > max_weight or total_space > max_space:
        return 0, total_weight, []  # Return 0 fitness if violated, along with total weight and empty list of packed items
    else:
        packed_items = [index for index, item in enumerate(individual) if item == 1]  # Get indices of packed items
        return total_value, total_weight, packed_items  # Otherwise, return the total value, total weight, and indices of packed items

# Selection - Best Two Parents
def selection(population, fitness_scores):
    # Sort the population based on fitness scores in descending order
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
    # Select the best two individuals as parents
    parent1 = sorted_population[0]
    parent2 = sorted_population[1]
    return parent1, parent2

# Crossover - Single Point Crossover
def crossover(parent1, parent2):
    # Randomly select a crossover point
    crossover_point = np.random.randint(len(parent1))
    # Perform single-point crossover
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Mutation
def mutation(individual):
    # Perform mutation on individual with a probability mutation_rate
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = 1 - individual[i]  # Flip the bit
    return individual

# Genetic Algorithm
def genetic_algorithm():
    # Initialize variables to keep track of the best fitness and generation
    best_fitness = 0
    best_generation = 0
    best_weight = 0
    best_packed_items = []
    fitness_history = []  # List to store best fitness for each generation
    avg_fitness_history = []  # List to store average fitness for each generation
    
    # Initialize the population
    population = initialize_population(population_size)
    
    # Loop through generations
    for generation in range(generations):
        # Calculate fitness scores for each individual in the population
        fitness_scores = [calculate_fitness(individual) for individual in population]
        
        # Find the best fitness score in the current generation
        current_best_fitness, current_best_weight, current_best_packed_items = max(fitness_scores)
        
        # Update the best fitness and generation if a better fitness is found
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_generation = generation
            best_weight = current_best_weight
            best_packed_items = current_best_packed_items
            
        # Calculate average fitness for the current generation
        avg_fitness = np.mean([score[0] for score in fitness_scores])
        
        # Store best and average fitness for this generation
        fitness_history.append(current_best_fitness)
        avg_fitness_history.append(avg_fitness)
        
        # Select the best individual from the current population
        best_individual_index = np.argmax([score[0] for score in fitness_scores])
        new_population = [population[best_individual_index]]
        
        # Selection, crossover, and mutation
        selected_population = selection(population, [score[0] for score in fitness_scores])
        for _ in range(int(population_size / 2)):
            parent1, parent2 = selected_population
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([child1, child2])
        mutated_population = [mutation(individual) for individual in new_population]
        population = mutated_population
        
        # Print generation and best fitness for monitoring
        print("Generation:", generation, "Best Fitness:", current_best_fitness, "Weight:", current_best_weight, "Packed Items:", current_best_packed_items)
        
    # Return the best overall fitness, the generation where it was achieved, fitness history, best weight, and packed items
    return best_fitness, best_generation, fitness_history, avg_fitness_history, best_weight, best_packed_items

# Plot fitness over generations
def plot_fitness_over_generations(fitness_history, avg_fitness_history):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(fitness_history) + 1), fitness_history, label='Best Fitness', marker='o', linestyle='-')
    plt.plot(range(1, len(avg_fitness_history) + 1), avg_fitness_history, label='Average Fitness', marker='x', linestyle='--')
    plt.title('Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.show()

''' Main loop '''
def main():
    # Run the genetic algorithm
    best_fitness, best_generation, fitness_history, avg_fitness_history, best_weight, best_packed_items = genetic_algorithm()

    # Print the best overall fitness, the generation where it was achieved, and best weight
    print("Best overall fitness:", best_fitness)
    print("Achieved at generation:", best_generation)   
    print("Weight:", best_weight)
    print("Packed items indices:", best_packed_items)
    
    # Plot fitness over generations
    plot_fitness_over_generations(fitness_history, avg_fitness_history)

if __name__ == "__main__":
    main()