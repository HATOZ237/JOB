----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 10/13/2021 10:46:53 AM
-- Design Name: 
-- Module Name: adder_16bits - Behavioral
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
library logic_com;
use logic_com.ALL;

-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
--use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx leaf cells in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity adder_16bits is
    Port ( A : in STD_LOGIC_VECTOR (15 downto 0);
           B : in STD_LOGIC_VECTOR (15 downto 0);
           output : out STD_LOGIC_VECTOR (15 downto 0));
end adder_16bits;

architecture Behavioral of adder_16bits is

signal se: STD_LOGIC_VECTOR (15 downto 0); 
signal sw: STD_LOGIC_VECTOR (15 downto 0); 
signal si: STD_LOGIC_VECTOR (15 downto 0); 

begin

adder : for i in 0 to 15 generate
    first_add: if (i = 0) generate
        add_0: entity logic_com.add
            port map(a=> A(0), b=> B(0), s=>sw(0), ci => '0', c0=>se(0));
        end generate;
    other_add: if (i>0) generate
        add_i: entity logic_com.add
                port map(a=> A(i), b=> B(i), s=>sw(i), ci => si(i-1), c0=>se(i));
            end generate;
end generate;
si<= se;
output <= sw;
end Behavioral;
