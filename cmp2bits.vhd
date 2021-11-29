----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 12.10.2021 21:08:31
-- Design Name: 
-- Module Name: cmp2bits - Behavioral
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

entity cmp2bits is
    Port ( A : in STD_LOGIC;
           B : in STD_LOGIC;
           cmp_i : in STD_LOGIC;
           cmp_o : out STD_LOGIC);
end cmp2bits;

architecture Behavioral of cmp2bits is

begin
cmp_o <= ((A or not B) and cmp_i) or (A and not B); 
end Behavioral;
