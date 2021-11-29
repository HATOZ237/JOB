----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 12.10.2021 22:34:26
-- Design Name: 
-- Module Name: compteur_16bits - Behavioral
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

entity compteur_16bits is
Port ( reset : in STD_LOGIC;
           --i_a : in std_logic_vector(7 downto 0);
           --i_b : in std_logic_vector(7 downto 0) := (others => '1');
           output: out std_logic_vector(15 downto 0);
           clk : in STD_LOGIC;
           enable : in STD_LOGIC);
end compteur_16bits;

architecture Behavioral of compteur_16bits is
signal sin: std_logic_vector(15 downto 0);
signal sout: std_logic_vector(15 downto 0) ;

signal sen: std_logic_vector(15 downto 0) := (others => '0');
--signal str: std_logic_vector(7 downto 0);

signal s_b: std_logic_vector(15 downto 0) := (0 => '1',others => '0');
begin

registre : for i in 0 to 15 generate
    reg_i : entity Logic_com.registre_1
        port map(rst => reset, en=> enable, clk=>clk, d=> sin(i), q=>sout(i));
end generate;

adder_16 : entity Logic_com.adder_16bits
port map(A => SEN, B=>s_B, output=>sin);


process(reset, clk)
begin
if(reset = '1') then
     sen <= (others => '0');
elsif(reset = '0') then
sen <= sout;
--output <= sout;

end if;
end process;
output <= sout;
end Behavioral;
