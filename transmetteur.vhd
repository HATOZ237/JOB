----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 10/21/2021 11:44:03 AM
-- Design Name: 
-- Module Name: transmetteur - Behavioral
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

entity transmetteur is
    Port ( start : in STD_LOGIC;
           
           reset : in STD_LOGIC;
           occupe : out STD_LOGIC;
           termine : out STD_LOGIC;
           datain : in STD_LOGIC_VECTOR (5 downto 0);
           clk : in std_logic;
           tx: out std_logic;
           rx: in std_logic);
end transmetteur;

architecture Behavioral of transmetteur is

signal data_int: std_logic_vector(7 downto 0);
signal tx_int : std_logic;
signal data_out: STD_LOGIC_VECTOR (7 downto 0);
signal data_rdy : std_logic;
signal start_t : std_logic;

begin

trans_uart: entity logic_com.Transmetteur_UART
port map (reset => reset,
          clk => clk,
          occupe => occupe,
          termine => termine,
          start => start,
          datain => data_int,
          tx => tx);
          
--recept_uart : entity logic_com.UART
--port map(reset=> reset,
--        clk => clk,
--        data_out => data_out,
--        rx => rx,
--        data_rdy => data_rdy);


data_int <= x"55";
--tx<= tx_int;
--process(data_rdy)
--begin
--if(data_rdy = '1') then 
--    start_t <= '1';
--elsif(data_rdy = '0') then
--    start_t <= '0';
    
--end if;
--end process;
end Behavioral;
