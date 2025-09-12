"""
File containing logic of Graphals group, to be loaded into the notebooks
"""

TIME_AT_PEAK_MAX_s = 6*60*60 #time a node is at most at peak production in seconds

def get_leaves(network):
    """
    Function to get the customers that are leaves i.e. no nodes behind
    """
    raise NotImplementedError

    return



def battery_allocation(network):
    """
    Function to generate the battery allocation for a LEG=Station

    Returns a battery for each customer, None if the customer should not have a battery

    network ... graph representation of the network
    """
    def _recursive_part(network):
        """recursive function, private"""
        leaves = get_leaves(network)
        batteries = dict()
        # go through leaves and do the same operation everywhere
        for leave in leaves:
            # get values
            preak_prod = get_peak_prod(leave) #peak production
            outgoing_power_capacity = get_outgoing_capacity(leave) #get outgoing capacity

            # init now battery
            bat_cap = 0
            bat_power = 0
            add_battery = False
            increase_production = peak_prod #delta to increase production at parent once done

            """"
            NOTE

            Could also try first improving the limiting line and comparing it to the cost of a battery
            """

            # potentially add battery
            if preak_prod > outgoing_power_capacity:
                add_battery = True
                #place battery if can produce more (this assumes that the consumption is 0 as we are looking at the worst case)
                emax = peak_prod*TIME_AT_PEAK_MAX_s #maximal energy that this node can give
                bat_cap = emax #can swallow worst case (assumes the battery does not discharge during it, can potentially lower)
                bat_power = emax / (24*60*60) #power discharge s.t. battery empties over a day (is a lower bound theoretically)

            
            # add to list
            batteries[leave] = SimpleBattery(capacity_kwh=bat_cap, max_power_kw=bat_cap)

            # write outgoing power as incoming power to parent
            parent = get_parent(leave)
            if add_battery:
                increase_production = bat_power
            parent.increase_production(increase_production) #increase the proudction at the parent, moving all information to the parent

        # remove leaves
        remove_leaves(network)

        return batteries

    # start 
    batteries = dict()
    local_batteries = _recursive_part(network)
    batteries.update(local_batteries) #extend the dictionary

    return batteries

def create_schedule(batteries_dict):
    """
    Function to creata a battery schedule given the battery at each customer

    """
    return


### FROM NOTEBOOK
class SimpleBattery:
    """A simple battery model that simulates energy storage with constraints."""
    def __init__(self, capacity_kwh, max_power_kw, efficiency=0.9, initial_soc_percent=5.0):
        self.capacity_kwh = float(capacity_kwh)
        self.max_power_kw = float(max_power_kw)
        self.efficiency = float(efficiency)
        self.soc_kwh = self.capacity_kwh * (initial_soc_percent / 100.0)

    def charge(self, power_kw, duration_hours):
        """Charges the battery, returning the actual power used after constraints."""
        power_to_charge = min(power_kw, self.max_power_kw)
        available_capacity_kwh = self.capacity_kwh - self.soc_kwh
        max_energy_in_kwh = available_capacity_kwh / self.efficiency
        max_power_for_duration = max_energy_in_kwh / duration_hours
        actual_power_in = min(power_to_charge, max_power_for_duration)
        energy_added_kwh = actual_power_in * duration_hours * self.efficiency
        self.soc_kwh += energy_added_kwh
        return actual_power_in

    def discharge(self, power_kw, duration_hours):
        """Discharges the battery, returning the actual power supplied after constraints."""
        power_to_discharge = min(power_kw, self.max_power_kw)
        max_power_for_duration = self.soc_kwh / duration_hours
        actual_power_out = min(power_to_discharge, max_power_for_duration)
        energy_removed_kwh = actual_power_out * duration_hours
        self.soc_kwh -= energy_removed_kwh
        return actual_power_out