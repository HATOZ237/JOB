----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 10/14/2021 11:28:18 AM
-- Design Name: 
-- Module Name: Transmetteur_UART - Behavioral
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
use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx leaf cells in this code.
library UNISIM;
use UNISIM.VComponents.all;

entity Transmetteur_UART is
    Port ( clk : in STD_LOGIC;
           start : in STD_LOGIC;
           reset : in STD_LOGIC;
           tx : out STD_LOGIC;
           termine : out STD_LOGIC;
           occupe : out STD_LOGIC;
           datain : in STD_LOGIC_VECTOR (7 downto 0)
          );
end Transmetteur_UART;



architecture Behavioral of Transmetteur_UART is
type etat is (attente, chargement, demarrage ,first_bit,  send_bit, end_bit, end_all);
signal etat_present : etat;
signal data_buffer: std_logic_vector(7 downto 0);

--signal enable_reg_dec: std_logic;
signal enable_comp_0: std_logic := '1';
signal out_comp_0: std_logic_vector(15 downto 0);

signal enable_comp_1: std_logic:= '1';
signal out_comp_1: std_logic_vector(15 downto 0);

signal reset_comp_0: std_logic := '1';
signal reset_comp_1: std_logic := '1';


signal cmp_0: std_logic;

signal cmp_1: std_logic;

--signal cmp_2: std_logic;

signal one: std_logic := '1';

signal data_rdy: STD_LOGIC;
signal enable_rdc: STD_LOGIC;
signal read : std_logic;
signal mode :  std_logic;
signal first_arrival :  std_logic;

signal input_data: STD_LOGIC_VECTOR (7 downto 0);

signal NBRE_COUP_HORLOGE: STD_LOGIC_VECTOR (15 downto 0):=    x"0001";  --"0000010000111101";--;
--signal HALF_NBRE_COUP_HORLOGE: STD_LOGIC_VECTOR (15 downto 0):= "0000001000011110";
signal Nbre_bits: STD_LOGIC_VECTOR (15 downto 0):=             "0010101001100101";--x"0001";
signal compte : integer:= 0;
signal ctrl_mux: std_logic;
signal out_tx: std_logic;

begin

comparateur_0: entity logic_com.cmp_16bits
    port map(a => out_comp_0, b => NBRE_COUP_HORLOGE, cmp => cmp_0);

--comparateur_1: entity logic_com.cmp_16bits
 --   port map(a => out_comp_0, b => HALF_NBRE_COUP_HORLOGE,cmp => cmp_1);

comparateur_1: entity logic_com.cmp_16bits
    port map(a => out_comp_1, b => Nbre_bits, cmp => cmp_1);
    
compteur_0: entity logic_com.compteur_16bits
    port map(reset => reset_comp_0, clk => clk, enable =>enable_comp_0, output =>out_comp_0 );

compteur_1: entity logic_com.compteur_16bits
    port map(reset => reset_comp_1, clk => clk, enable =>enable_comp_1, output =>out_comp_1 );
    
registre_dec: entity logic_com.rdc_load_Nbits
    generic map(N =>8)
    port map(reset =>reset, mode =>mode, load => input_data, input =>'1', output => read, clk=>clk, enable => enable_rdc );
    
mux : entity mux_2_1
port map(ctrl => ctrl_mux,
           in_value => read,
           in_pos => '1',
           output =>out_tx);


process(reset, start, clk, mode, datain)
begin
if(reset = '1') then
    etat_present <= attente;
    ctrl_mux <= '1';
    tx <= out_tx;
    first_arrival <= '1';
elsif(rising_edge(clk)) then
    case(etat_present) is
        when attente =>
            first_arrival <= '1';
            tx <= out_tx;
            ctrl_mux <= '1';
            occupe <='0';
            termine <= '0';
            enable_rdc <= '0';
            reset_comp_0 <= '1';
            reset_comp_1 <= '1';
            if(start = '0') then 
                etat_present <= attente;
            elsif(start = '1') then
                etat_present <= chargement;
            end if;
        
        when chargement =>
            first_arrival <= '1';
            tx <= out_tx;
            ctrl_mux <= '1';
            occupe <='0';
            termine <= '0';
            input_data <= datain;
            reset_comp_0 <= '1';
            reset_comp_1 <= '1';
            etat_present <= demarrage;
            enable_rdc <= '1';
            mode <= '0';
            
        when demarrage =>
            first_arrival <= '1';
            tx <= out_tx;
            ctrl_mux <= '1';
            occupe <='0';
            termine <= '0';
            enable_rdc <= '0';
            reset_comp_0 <= '1';
            reset_comp_1 <= '1';
            mode <= '1';
            etat_present <= first_bit;
            
        when first_bit =>
            first_arrival <= '1';
            ctrl_mux <= '0';
            tx <= '0';
            reset_comp_0 <= '0';
            reset_comp_1 <= '1';
            if( cmp_0 = '1') then
                reset_comp_0 <= '1';
                etat_present <= send_bit;
                --enable_rdc <= '1';
            elsif(cmp_0 = '0') then
                etat_present <= first_bit;
            end if;
            occupe <='1';
            termine <= '0';
            
            
        when send_bit =>
            --ctrl_mux <= '0';
--            if(first_arrival = '1') then
--                enable_rdc <= '1';
--                first_arrival <= '0';
--            elsif(first_arrival = '0') then 
--                enable_rdc <= '0';
--            end if;
            reset_comp_1 <= '0';
            reset_comp_0 <= '0';
            occupe <='1';
            termine <= '0';
            enable_rdc <= '0';
            tx<= out_tx;
            if(compte = 9) then
                etat_present <= end_bit;
                compte <= 0;
                --enable_rdc <= '1';
            else
                if(cmp_0 = '0') then
                    etat_present <= send_bit;
                elsif(cmp_0 = '1') then
                    etat_present <= send_bit;
                    enable_rdc <= '1';
                    reset_comp_0 <= '1';
                    compte <= compte +1;
                end if;
            end if;
            
            
        
        when end_bit =>
            --ctrl_mux <= '0';
            tx <= '1';
            reset_comp_0 <= '0';
            reset_comp_1 <= '1';
            if( cmp_0 = '1') then
                reset_comp_0 <= '1';
                etat_present <= end_all;
                --enable_rdc <= '1';
            elsif(cmp_0 = '0') then
                etat_present <= end_bit;
            end if;
            occupe <='1';
            termine <= '0';
        
        when end_all =>
            ctrl_mux <= '1';
            occupe <='0';
            termine <= '1';
            enable_rdc <= '0';
            reset_comp_0 <= '1';
            reset_comp_1 <= '1';
            occupe <='0';
            termine <= '1';
            etat_present <= attente;
    end case;
end if;
end process;
end Behavioral;
