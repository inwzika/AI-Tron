# -*- coding: utf-8 -*-

# python imports

import copy
import random

# chillin imports
from chillin_client import RealtimeAI

# project imports
from ks.models import ECell, EDirection, Agent, World
from ks.commands import ChangeDirection, ActivateWallBreaker
import time
import numpy as np
from tqdm import tqdm
import itertools


class MoveNode:
    def __init__(self, move):
        self.move = move
        self.children = {}

    def add_child(self, move):
        if not self.children.get(move):
            self.children[move] = MoveNode(move)
        return self.children[move]


def check_move(direction, current_player, world):
    if direction == EDirection.Right and current_player.position.x + 1 < len(world.board[0]):
        return True
    elif direction == EDirection.Left and current_player.position.x - 1 >= 0:
        return True
    elif direction == EDirection.Down and current_player.position.y + 1 < len(world.board):
        return True
    elif direction == EDirection.Up and current_player.position.y - 1 >= 0:
        return True
    return False


def final_check_move(move, current_player, world):
    if move == "ActivateWallBreaker" and (
            current_player.wall_breaker_cooldown != 0 or current_player.wall_breaker_rem_time != 0 or not check_move(
        current_player.direction,
        current_player, world)):
        return False
    if move == "ActivateWallBreaker":
        return True
    if (current_player.direction.value + 2) % 4 == move.value or not check_move(move,
                                                                                current_player, world):
        return False

    return True


def crossover(parent1, parent2):
    # Perform one-point crossover
    crossover_point = random.randint(0, len(parent1))
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def selection(population, fitness_scores, selection_size, randomized=False):
    selected_population = []
    selected_fitnesses = []
    total_fitness = sum(fitness_scores)
    if total_fitness == 0 or randomized:
        return random.sample(population, selection_size)
    probabilities = [fitness / total_fitness for fitness in fitness_scores]
    cumulative_probabilities = [sum(probabilities[:i + 1]) for i in range(len(probabilities))]

    while len(selected_population) < selection_size:
        rand_num = random.random()
        for i, cumulative_prob in enumerate(cumulative_probabilities):
            if rand_num <= cumulative_prob:
                selected_population.append(population[i])
                selected_fitnesses.append(fitness_scores[i])
                break

    return selected_population, selected_fitnesses


def mutation(individual):
    # Perform one-point mutation
    mutation_point = random.randint(0, len(individual) - 1)
    individual[mutation_point] = random.randint(0, 4)
    return individual


def get_tree(population):
    root = MoveNode(None)
    for j in range(len(population)):
        last_node = root
        for i, move in enumerate(population[j]):
            last_node = last_node.add_child(move)
    return root


def int_to_move(move):
    if move < 4:
        return list(EDirection)[move]
    elif move == 4:
        return "ActivateWallBreaker"
    pass


def calculate_fitness(best_moves, population):
    scores = [1] * len(population)
    for i, individual in enumerate(population):
        for j, move in enumerate(individual):
            if move == best_moves[j]:
                scores[i] += 1
            else:
                break
    return scores


def get_area_moves(x, y, world):
    if x <= len(world.board[0]) / 2 and y <= len(world.board) / 2:
        if x <= 2:
            return [EDirection.Down, EDirection.Left, EDirection.Right, EDirection.Up]
        return [EDirection.Left, EDirection.Down, EDirection.Right, EDirection.Up]
    if x <= len(world.board[0]) / 2 and y > len(world.board) / 2:
        if y >= len(world.board) - 3:
            return [EDirection.Right, EDirection.Down, EDirection.Up, EDirection.Left]
        return [EDirection.Down, EDirection.Right, EDirection.Up, EDirection.Left]
    if x > len(world.board[0]) / 2 and y > len(world.board) / 2:
        if y >= len(world.board) - 3:
            return [EDirection.Up, EDirection.Right, EDirection.Left, EDirection.Down]
        return [EDirection.Right, EDirection.Up, EDirection.Left, EDirection.Down]
    if y <= 2:
        return [EDirection.Left, EDirection.Up, EDirection.Down, EDirection.Right]
    return [EDirection.Up, EDirection.Left, EDirection.Down, EDirection.Right]


class AI(RealtimeAI):

    def __init__(self, world):
        super(AI, self).__init__(world)
        self.all_moves_other_player = None
        self.all_moves_my_player = None
        self.cycle = None
        self.ENEMY_WALL = None
        self.MY_WALL = None
        self.Health_decreasing_score = None
        self.population_size = 5000
        self.selection_size = 200
        self.num_generations = 1
        self.strategy_length = 6
        self.mutation_rate = 0.2
        self.last_decide = None
        self.first_population = [[], [], [], []]

    def first_population_init(self):
        numbers = [1, 2, 3, 4, 0]
        combinations = list(itertools.product(numbers, repeat=self.strategy_length))
        for elem in combinations:
            elem = list(elem)
            first_move = elem[0]
            my_dir = elem[0]
            other_dir = elem[1]
            my_wall = True
            other_wall = True
            flag_check = True
            for i, move in enumerate(elem):
                if move == 4:
                    if i % 2 == 0:
                        if not my_wall:
                            break
                        my_wall = False
                    else:
                        if not other_wall:
                            break
                        other_wall = False
                else:
                    if i % 2 == 0:
                        if (my_dir + 2) % 4 == move and my_dir != 4:
                            flag_check = False
                            break
                        my_dir = move
                    else:
                        if (other_dir + 2) % 4 == move and other_dir != 4:
                            flag_check = False
                            break
                        other_dir = move
            if flag_check and first_move != 4:
                self.first_population[(first_move + 1) % 4].append(elem)
                self.first_population[(first_move + 3) % 4].append(elem)
                self.first_population[(first_move + 0) % 4].append(elem)
            elif flag_check:
                for i in range(4):
                    self.first_population[i].append(elem)

    def initialize(self):
        start_time = time.time()
        self.Health_decreasing_score = int(
            (self.world.constants.enemy_wall_crash_score + self.world.constants.my_wall_crash_score)
            / (2 * self.world.constants.init_health))
        print(self.Health_decreasing_score)
        if self.my_side == "Yellow":
            self.MY_WALL = ECell.YellowWall
            self.ENEMY_WALL = ECell.BlueWall
        else:
            self.MY_WALL = ECell.BlueWall
            self.ENEMY_WALL = ECell.YellowWall
        self.cycle = 0
        self.first_population_init()
        print(len(self.first_population[0]))
        # current_my_direction_value = self.world.agents[self.my_side].direction.value
        # self.all_moves = [self.world.agents[self.my_side].direction,
        #                   list(EDirection)[(current_my_direction_value + 2) % 4],
        #                   list(EDirection)[(current_my_direction_value + 1) % 4],
        #                   list(EDirection)[(current_my_direction_value + 3) % 4],
        #                   "ActivateWallBreaker"]

        print("--- initialize %s seconds ---" % (time.time() - start_time))

    def decide(self):
        start_time = time.time()
        best_movement = self.genetic(world=self.world, cycle=self.cycle)
        self.last_decide = best_movement
        print("Best Move: ", best_movement)
        print("#######################################################")
        if best_movement == "ActivateWallBreaker":
            self.send_command(ActivateWallBreaker())
        else:
            self.send_command(ChangeDirection(best_movement))
        self.cycle += 1
        print("--- decide %s seconds ---" % (time.time() - start_time))

    def initialize_population(self, my_direction=None, other_direction=None):
        # population = []
        # for _ in range(self.population_size):
        #     player_strategy = []
        #     for i in range(self.strategy_length):
        #         rand_num = random.choice(self.get_possible_moves(player_strategy, my_direction, other_direction))
        #         tmp = 0
        #         while rand_num == 4 and tmp <= self.selection_size / 2:
        #             tmp += 1
        #             rand_num = random.choice(self.get_possible_moves(player_strategy, my_direction, other_direction))
        #         player_strategy.append(rand_num)
        #     population.append(player_strategy)
        population = random.sample(self.first_population[my_direction], self.population_size)
        return population

    def minimax(self, current_node: MoveNode, world: World, depth: int, my_turn, cycle, best_moves
                ):

        current_player, other_player, my_side, other_side = self.get_players(my_turn, world)
        if depth == 0 or self.check_end_game(world, cycle=cycle):
            # print(self.get_score(world))
            # print("(", current_player.position.x, ",", current_player.position.y, ")")
            # print("(", other_player.position.x, ",", other_player.position.y, ")")
            # print("---------------------------")
            return self.get_score(world)
        all_moves = get_area_moves(current_player.position.x, current_player.position.y, world)
        all_moves.append("ActivateWallBreaker")
        if my_turn:
            value = float('-inf')
            for move in all_moves:
                if move == "ActivateWallBreaker":
                    int_move = 4
                else:
                    int_move = move.value
                if int_move in current_node.children.keys():
                    # move = int_to_move(int_move)
                    # print("Me", move, ", ", self.strategy_length - depth)
                    if final_check_move(move, current_player, world):
                        child = self.handle_new_state(world, True, move)
                        tmp = self.minimax(current_node.children[int_move], child, depth - 1, False, cycle, best_moves)
                        del child
                        if tmp > value:
                            value = tmp
                            best_movement = int_move
                            best_moves[self.strategy_length - depth] = best_movement
        else:
            value = float('inf')
            for move in all_moves:
                if move == "ActivateWallBreaker":
                    int_move = 4
                else:
                    int_move = move.value
                if int_move in current_node.children.keys():
                    # move = int_to_move(int_move)
                    # print("You", move, ", ", self.strategy_length - depth)
                    if final_check_move(move, current_player, world):
                        child = self.handle_new_state(world, False, move)
                        tmp = self.minimax(current_node.children[int_move], child, depth - 1, True, cycle + 1,
                                           best_moves)
                        del child
                        if tmp < value:
                            value = tmp
                            best_movement = int_move
                            best_moves[self.strategy_length - depth] = best_movement

        return value

    def print_pop(self, population, fitness_scores):
        for i in range(len(population)):
            print(fitness_scores[i], [int_to_move(elem) for elem in population[i]])
        print("______________________________________")

    def genetic(self, world: World, cycle):
        population = self.initialize_population(world.agents[self.my_side].direction.value,
                                                world.agents[self.other_side].direction.value)
        # for i in range(len(population)):
        #     print([int_to_move(elem) for elem in population[i]])
        # print("#################################################")
        for generation in range(self.num_generations):
            # tree = get_tree(population)
            # best_moves = [None] * self.strategy_length
            # self.minimax(tree, world, self.strategy_length, True, cycle, best_moves)
            # fitness_scores = self.calculate_fitness(best_moves, population)

            # Select the individuals for reproduction based on fitness
            selected_population = selection(population, [], self.selection_size, randomized=True)
            # Create the next generation through crossover and mutation
            next_generation = population
            epoch = 0
            while epoch < self.selection_size / 2:
                parent1, parent2 = random.sample(selected_population, 2)
                child1, child2 = crossover(parent1, parent2)
                if random.random() < self.mutation_rate:
                    child1 = mutation(child1)
                if random.random() < self.mutation_rate:
                    child2 = mutation(child2)
                next_generation.append(child1)
                next_generation.append(child2)
                epoch += 1

            tree = get_tree(next_generation)
            best_moves = [None] * self.strategy_length
            self.minimax(tree, world, self.strategy_length, True, cycle, best_moves)
            fitness_scores = calculate_fitness(best_moves, next_generation)
            # Replace the current population with the next generation
            population, fitness_scores = selection(next_generation, fitness_scores, self.population_size)
            # self.print_pop(next_generation, fitness_scores)

        best_elem = population[np.argmax(fitness_scores)]
        # self.print_pop(population, fitness_scores)
        # print(np.max(fitness_scores) - 1, [int_to_move(elem) for elem in best_elem])
        # print("AVG : ", (np.sum(fitness_scores) - self.population_size) / self.population_size)
        # print(population)
        return int_to_move(best_elem[0])

    def handle_new_state(self, world: World, my_turn: bool, move):
        new_world = copy.deepcopy(world)

        current_player, other_player, my_side, other_side = self.get_players(my_turn, new_world)
        first_move = move
        if move == "ActivateWallBreaker":
            # new_world.scores[my_side] -= new_world.constants.wall_score_coefficient * 3
            current_player.wall_breaker_rem_time = new_world.constants.wall_breaker_duration
            move = current_player.direction
        if move == EDirection.Right:
            current_player.position.x += 1
        elif move == EDirection.Left:
            current_player.position.x -= 1
        elif move == EDirection.Up:
            current_player.position.y -= 1
        elif move == EDirection.Down:
            current_player.position.y += 1
        current_player.direction = move

        if my_turn:
            my_wall = self.MY_WALL
            enemy_wall = self.ENEMY_WALL
        else:
            my_wall = self.ENEMY_WALL
            enemy_wall = self.MY_WALL
        if new_world.board[other_player.position.y][other_player.position.x] != ECell.AreaWall:
            new_world.board[other_player.position.y][other_player.position.x] = enemy_wall
        current_map_position = new_world.board[current_player.position.y][current_player.position.x]

        if current_player.wall_breaker_rem_time > 0:
            current_player.wall_breaker_rem_time -= 1
            if current_player.wall_breaker_rem_time == 0:
                current_player.wall_breaker_cooldown = new_world.constants.wall_breaker_cooldown

        activated_time_wall = new_world.constants.wall_breaker_duration - current_player.wall_breaker_rem_time
        if current_player.wall_breaker_rem_time == 0 or activated_time_wall <= 2:
            activated_time_wall = 0

        # activated_time_wall = 0  # TODO
        if current_map_position == ECell.Empty:
            if first_move == "ActivateWallBreaker":
                new_world.scores[my_side] -= new_world.constants.wall_score_coefficient * 3
            new_world.scores[my_side] += new_world.constants.wall_score_coefficient
            new_world.board[current_player.position.y][current_player.position.x] = my_wall


        elif current_map_position == my_wall:
            new_world.scores[my_side] -= new_world.constants.wall_score_coefficient * activated_time_wall
            if current_player.wall_breaker_rem_time == 0:
                current_player.health -= 1
                new_world.scores[my_side] += self.Health_decreasing_score
            if current_player.health == 0:
                new_world.scores[my_side] += new_world.constants.my_wall_crash_score

        elif current_map_position == enemy_wall:
            new_world.scores[my_side] += new_world.constants.wall_score_coefficient
            new_world.scores[my_side] -= new_world.constants.wall_score_coefficient * activated_time_wall
            new_world.scores[other_side] -= new_world.constants.wall_score_coefficient
            if current_player.wall_breaker_rem_time == 0:
                current_player.health -= 1
                new_world.scores[my_side] += self.Health_decreasing_score
            if current_player.health == 0:
                new_world.scores[my_side] += new_world.constants.enemy_wall_crash_score
            new_world.board[current_player.position.y][current_player.position.x] = my_wall

        elif current_map_position == ECell.AreaWall:
            new_world.scores[my_side] += new_world.constants.area_wall_crash_score
            new_world.scores[my_side] += self.Health_decreasing_score * current_player.health
            current_player.health = 0

        if current_player.position.x == other_player.position.x and current_player.position.y == other_player.position.y:
            if not my_turn:
                current_player.health = 0
                other_player.health = 0
                new_world.scores[self.my_side] -= new_world.constants.wall_score_coefficient * 4
                # print(-self.get_score(world))
                # print("(", current_player.position.x, ",", current_player.position.y, ")")
                # print("(", other_player.position.x, ",", other_player.position.y, ")")
        # update_time_vars
        if current_player.wall_breaker_cooldown > 0:
            current_player.wall_breaker_cooldown -= 1
        return new_world

    def get_score(self, world: World):
        current_player, other_player, current_side, other_side = self.get_players(True, world)
        # return (world.scores[current_side] - world.scores[
        #     other_side]) * 2 - current_player.wall_breaker_cooldown + other_player.wall_breaker_cooldown
        return world.scores[current_side] - world.scores[other_side]

    def get_players(self, my_turn, new_world):
        if my_turn:
            current_player: Agent = new_world.agents[self.my_side]
            other_player: Agent = new_world.agents[self.other_side]
            my_side = self.my_side
            other_side = self.other_side
        else:
            current_player: Agent = new_world.agents[self.other_side]
            other_player: Agent = new_world.agents[self.my_side]
            other_side = self.my_side
            my_side = self.other_side
        return current_player, other_player, my_side, other_side

    def check_end_game(self, world: World, cycle: int) -> bool:
        if world.agents[self.other_side].health == 0 or world.agents[self.my_side].health == 0:
            return True
        if cycle >= world.constants.max_cycles:
            return True
        return False

    ##deprecated
    def get_possible_moves(self, last_strategy, my_direction=None, other_direction=None):
        if len(last_strategy) == 0:
            if self.last_decide == "ActivateWallBreaker":
                return [my_direction, (my_direction + 1) % 4, (my_direction + 3) % 4]
            else:
                return [my_direction, (my_direction + 1) % 4, (my_direction + 3) % 4, 4]
        if len(last_strategy) == 1:
            return [other_direction, (other_direction + 1) % 4, (other_direction + 3) % 4, 4]
        if len(last_strategy) == 2:
            if last_strategy[0] == 4:
                last_move = my_direction
                return [last_move, (last_move + 1) % 4, (last_move + 3) % 4]
            else:
                last_move = last_strategy[0]
                return [last_move, (last_move + 1) % 4, (last_move + 3) % 4, 4]
        if len(last_strategy) == 3:
            if last_strategy[1] == 4:
                last_move = other_direction
                return [last_move, (last_move + 1) % 4, (last_move + 3) % 4]
            else:
                last_move = last_strategy[1]
                return [last_move, (last_move + 1) % 4, (last_move + 3) % 4, 4]
        if last_strategy[-2] == 4:
            last_move = last_strategy[-4]
            return [last_move, (last_move + 1) % 4, (last_move + 3) % 4]
        else:
            last_move = last_strategy[-2]
            return [last_move, (last_move + 1) % 4, (last_move + 3) % 4, 4]
