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


def get_area_moves(x, y, world):
    if x <= len(world.board[0]) / 2 and y <= len(world.board) / 2:
        if x == 2:
            return [EDirection.Down, EDirection.Left, EDirection.Right, EDirection.Up]
        return [EDirection.Left, EDirection.Down, EDirection.Right, EDirection.Up]
    if x <= len(world.board[0]) / 2 and y > len(world.board) / 2:
        if y == len(world.board) - 3:
            return [EDirection.Right, EDirection.Down, EDirection.Up, EDirection.Left]
        return [EDirection.Down, EDirection.Right, EDirection.Up, EDirection.Left]
    if x > len(world.board[0]) / 2 and y > len(world.board) / 2:
        if y == len(world.board) - 3:
            return [EDirection.Up, EDirection.Right, EDirection.Left, EDirection.Down]
        return [EDirection.Right, EDirection.Up, EDirection.Left, EDirection.Down]
    if y == 2:
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
        self.depth = 8

    def initialize(self):
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
        # current_my_direction_value = self.world.agents[self.my_side].direction.value
        # self.all_moves_my_player = [self.world.agents[self.my_side].direction,
        #                             list(EDirection)[(current_my_direction_value + 2) % 4],
        #                             list(EDirection)[(current_my_direction_value + 1) % 4],
        #                             list(EDirection)[(current_my_direction_value + 3) % 4]]
        #
        # current_other_direction_value = self.world.agents[self.other_side].direction.value
        # self.all_moves_other_player = [self.world.agents[self.other_side].direction,
        #                                list(EDirection)[(current_other_direction_value + 2) % 4],
        #                                list(EDirection)[(current_other_direction_value + 1) % 4],
        #                                list(EDirection)[(current_other_direction_value + 3) % 4]]

        print('initialize')

    def decide(self):
        start_time = time.time()
        value, best_movement = self.minimax(world=self.world, depth=self.depth, my_turn=True, cycle=self.cycle)
        print("Best Move: ", best_movement)
        print("#######################################################")
        # if self.cycle == 0:
        # current_my_direction_value = self.world.agents[self.my_side].direction.value
        # self.all_moves_my_player = [list(EDirection)[(current_my_direction_value + 0) % 4],
        #                             list(EDirection)[(current_my_direction_value + 2) % 4],
        #                             list(EDirection)[(current_my_direction_value + 1) % 4],
        #                             list(EDirection)[(current_my_direction_value + 3) % 4]]
        #
        # current_other_direction_value = self.world.agents[self.other_side].direction.value
        # self.all_moves_other_player = [list(EDirection)[(current_other_direction_value + 0) % 4],
        #                                list(EDirection)[(current_other_direction_value + 2) % 4],
        #                                list(EDirection)[(current_other_direction_value + 1) % 4],
        #                                list(EDirection)[(current_other_direction_value + 3) % 4]]
        if best_movement == "ActivateWallBreaker":
            self.send_command(ActivateWallBreaker())
        else:
            self.send_command(ChangeDirection(best_movement))
        self.cycle += 1
        print("--- decide %s seconds ---" % (time.time() - start_time))

    def minimax(self, world: World, depth: int, my_turn, cycle, alpha=float('-inf'), beta=float('inf'),
                ):

        current_player, other_player, my_side, other_side = self.get_players(my_turn, world)
        possible_moves = []

        if depth == 0 or self.check_end_game(world, cycle=cycle):
            # print(self.get_score(world))
            # print("(", current_player.position.x, ",", current_player.position.y, ")")
            # print("(", other_player.position.x, ",", other_player.position.y, ")")
            # print("---------------------------")
            return self.get_score(world), None
        all_moves = get_area_moves(current_player.position.x, current_player.position.y, world)
        for direction in all_moves:
            if (current_player.direction.value + 2) % 4 != direction.value and check_move(direction,
                                                                                          current_player, world):
                possible_moves.append(direction)
        # random.shuffle(possible_moves)  # TODO
        if current_player.wall_breaker_cooldown == 0 and current_player.wall_breaker_rem_time == 0 and check_move(
                current_player.direction,
                current_player, world):
            possible_moves.append("ActivateWallBreaker")
        if my_turn:
            value = float('-inf')
            for move in possible_moves:
                # print("Me", move, ", ", self.depth - depth)
                child = self.handle_new_state(world, True, move)
                tmp = self.minimax(child, depth - 1, False, cycle, alpha, beta)[0]
                del child
                if tmp > value:
                    value = tmp
                    best_movement = move
                if value >= beta:
                    break
                alpha = max(alpha, value)
        else:
            value = float('inf')
            for move in possible_moves:
                # print("You", move, ", ", self.depth - depth)
                child = self.handle_new_state(world, False, move)
                tmp = self.minimax(child, depth - 1, True, cycle + 1, alpha, beta)[0]
                del child
                if tmp < value:
                    value = tmp
                    best_movement = move
                if value <= alpha:
                    break
                beta = min(beta, value)

        return value, best_movement

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
