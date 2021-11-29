----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 09/16/2021 11:00:15 AM
-- Design Name: 
-- Module Name: registre_1 - Behavioral
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

entity registre_1 is
    Port ( D : in STD_LOGIC;
           RST : in STD_LOGIC;
           EN : in STD_LOGIC;
           Q : out STD_LOGIC;
           clk : in STD_LOGIC);
end registre_1;

architecture Behavioral of registre_1 is

begin

process(d, rst, clk)
begin 
if (rst = '1') then 
    q <= '0';
elsif(rising_edge(clk)) then  
    if (en = '1') then 
        q <= d;
    end if;
end if;
end process;


end Behavioral;
