----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 11/29/2021 01:01:47 PM
-- Design Name: 
-- Module Name: mux_2_1 - Behavioral
-- Project Name: 
-- Target Devices: 
-- Tool Versions: 
-- Description: 
-- 
-- Dependencies: 
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------


library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
--use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx leaf cells in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity mux_2_1 is
    Port ( ctrl : in STD_LOGIC;
           in_value : in STD_LOGIC;
           in_pos : in STD_LOGIC;
           output : out STD_LOGIC);
end mux_2_1;

architecture Behavioral of mux_2_1 is

begin
output <= in_value when ctrl = '0' else
          in_pos   when ctrl = '1' ;
          
            

end Behavioral;
