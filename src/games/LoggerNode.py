import numpy as np


class LoggerNode:
    """
    This class represents node of a logger.
    Main usage of this class is to save information that changes from action to action.

    Attributes:
        current_state (np.array[]): The current state of the game.
        current_player (int): The current player who should make an action.
        last_action (int): Index of last action made by the previous player. (-1 means there are now prev actions)
        additional_vars (np.array[]): Additional variables that required for games.
        parent (LoggerNode): The parent of this node.
        child (LoggerNode): The child of this node.
    """

    def __init__(self, current_state=None, current_player=1, last_action=-1, additional_vars=np.array([], dtype=object),
                 parent=None, child=None):
        """
        Constructor.

        Parameters:
            current_state (np.array[]): The current state of the game.
            current_player (int): The current player who should make an action.
            last_action (int): Index of last action made by the previous player. (-1 means there is no prev actions)
            additional_vars (np.array[]): Additional variables that required for games.
            parent (LoggerNode): The parent of this node.
            child (LoggerNode): The child of this node.
        """
        self.current_state = current_state
        self.current_player = current_player
        self.last_action = last_action
        self.additional_vars = additional_vars
        self.parent = parent
        self.child = child

    def delete_child(self):
        """
        Node for deleting a child of this node.
        """
        if self.child:
            self.child.parent = None
            self.child.child = None
