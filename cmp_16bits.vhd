----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 01.10.2021 18:25:15
-- Design Name: 
-- Module Name: cmp_16bits - Behavioral
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
use ieee.numeric_std.all;
library logic_com;
use logic_com.ALL;

-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
--use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx leaf cells in this code.
library UNISIM;
use UNISIM.VComponents.all;

entity cmp_16bits is
    Port ( A : in STD_LOGIC_VECTOR (15 downto 0);
           B : in STD_LOGIC_VECTOR (15 downto 0);
           CMP : out STD_LOGIC);
end cmp_16bits;

architecture Behavioral of cmp_16bits is

--signal s_1 : STD_LOGIC_VECTOR (15 downto 0);
--signal s_2 : STD_LOGIC_VECTOR (15 downto 0);
signal s_out : STD_LOGIC_VECTOR (15 downto 0);

begin


s_out(0) <= a(0) or not b(0);

comparator: for i in 1 to 15 generate
  Comp_0: entity logic_com.cmp2bits
    port map(a=> A(i),b=> b(i), cmp_i=> s_out(i-1),cmp_o=> s_out(i));
end generate;
    
cmp <= s_out(15);

end Behavioral;
