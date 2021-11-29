----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 09/24/2021 12:56:27 PM
-- Design Name: 
-- Module Name: add - Behavioral
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

entity add is
    Port ( a : in STD_LOGIC;
           b : in STD_LOGIC;
           ci : in std_logic;
           c0 : out STD_LOGIC;
           S : out STD_LOGIC);
end add;

architecture Behavioral of add is

begin
s <= a xor b xor ci;
c0 <= (a and b) or(ci and a) or (ci and b); 

end Behavioral;
